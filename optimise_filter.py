import sys
import tensorflow as tf
import gc
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, ResNet50, DenseNet121, MobileNetV2, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score, balanced_accuracy_score,
                            precision_score, recall_score, roc_curve, auc, confusion_matrix, classification_report)
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.cluster import KMeans
import os

tf.config.set_soft_device_placement(True)

def quantize_gabor_number(value,fractional_width):

    if(abs(value) == 1):
        return value

    b = ''
    
    if(value < 0):
        b = b + '10'
        value = -value
    else:
        b = b + '00'

    for i in range(1,fractional_width+1):
        value = value * 2
        if(value >= 1):
            b = b + '1'
            value = value - 1
        else:
            b = b + '0'
    
    fractional_value = 0.0
    for i in range(1, fractional_width+1):
        if(b[i+1] == '1'):
            fractional_value += 2 ** (-i)

    if b[0] == '1':
        fractional_value = -fractional_value
    
    return fractional_value

def get_modified_kernel(original_kernel,n_clusters,ksize):
    kernel_flat = original_kernel.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init=10)
    kmeans.fit(kernel_flat)
    centroids = kmeans.cluster_centers_
    new_kernel_flat = centroids[kmeans.labels_]
    return new_kernel_flat.reshape(ksize)

def get_modified_kernel_with_precision(original_kernel, n_clusters, ksize, bit_widths):
    kernel_flat = original_kernel.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init=10)
    kmeans.fit(kernel_flat)
    centroids = kmeans.cluster_centers_
    for i in range(n_clusters):
        centroids[i] = quantize_gabor_number(centroids[i],bit_widths[i])
    new_kernel_flat = centroids[kmeans.labels_]
    return new_kernel_flat.reshape(ksize)

@tf.keras.utils.register_keras_serializable()
class GaborFilterBank(tf.keras.layers.Layer):
    def __init__(self, 
                 num_filters=4, 
                 kernel_size=(5, 5), 
                 orientations=4, 
                 lambd=8, 
                 sigma=3.37305, 
                 gamma_init=1.0, 
                 psi_init=0.0,
                 theta=None,
                 bit_widths=None,
                 clusters=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.orientations = orientations
        self.lambd = lambd
        self.sigma = sigma
        self.gamma_init = gamma_init
        self.psi_init = psi_init
        self.bit_widths = bit_widths
        self.clusters = clusters

        if theta is None:
            self.theta = np.array([np.deg2rad(angle) for angle in [45, 80, 135, 180]], dtype=np.float32)
        else:
            if isinstance(theta, (list, tuple, np.ndarray)):
                assert len(theta) == num_filters, f"Theta list must be of length {num_filters}"
                self.theta = np.array(theta, dtype=np.float32)
            else:
                raise ValueError("theta must be a list, tuple, or ndarray of length equal to num_filters")
    
    def get_gabor_kernel(self, sigma, theta, lambd, gamma):
        """Creates a trainable Gabor kernel."""
        ksize = self.kernel_size
        psi = self.psi_init
        kernel = cv2.getGaborKernel(
            ksize=ksize,
            sigma=sigma,
            theta=theta,
            lambd=lambd,
            gamma=gamma,
            psi=psi,
            ktype=cv2.CV_32F
        )

        if self.bit_widths and not self.clusters:
            modified_kernel = np.zeros(ksize)
            for i in range(ksize[0]):
                for j in range(ksize[1]):
                    modified_kernel[i][j] = quantize_gabor_number(kernel[i][j],self.bit_widths[i][j])
            kernel = np.array(modified_kernel)
        
        if self.clusters:
            # kernel = get_modified_kernel_with_precision(kernel, self.clusters, ksize, self.bit_widths)
            kernel = get_modified_kernel(kernel,self.clusters,ksize)

        kernel = kernel.astype(np.float32)
        kernel = tf.convert_to_tensor(kernel, dtype=tf.float32)
        kernel = tf.reshape(kernel, list(self.kernel_size) + [1, 1])
        return kernel

    def get_final_gabor_kernel_from_best_params(self, best_params):
        """Returns the Gabor kernel created using the best parameters from Optuna."""
        sigma = best_params['sigma']
        theta = best_params['theta']
        lambd = best_params['lambd']
        gamma = best_params['gamma']
        gabor_kernel = self.get_gabor_kernel(sigma, theta, lambd, gamma)
        return gabor_kernel.numpy()

    def call(self, inputs):
        """Applying Gabor filter bank to input images."""
        inputs = tf.cast(inputs, tf.float32)
        filters = []
        for i in range(self.num_filters):
            kernel = self.get_gabor_kernel(
                self.sigma,
                self.theta[i],
                self.lambd,
                self.gamma_init
            )
            filters.append(kernel)
        gabor_filters = tf.concat(filters, axis=-1)
        filtered = tf.nn.conv2d(
            inputs, 
            gabor_filters, 
            strides=[1, 1, 1, 1], 
            padding='SAME'
        )
        return filtered

    def get_config(self):
        """Return the configuration of the layer for serialization."""
        config = super().get_config()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': list(self.kernel_size),  # Ensure tuple is serializable
            'orientations': self.orientations,
            'lambd': float(self.lambd),  # Ensure float is serializable
            'sigma': float(self.sigma),
            'gamma_init': float(self.gamma_init),
            'psi_init': float(self.psi_init),
            'theta': self.theta.tolist()  # Convert NumPy array to list
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create a layer from its configuration."""
        # Convert theta back to a NumPy array
        config['theta'] = np.array(config['theta'], dtype=np.float32)
        # Convert kernel_size back to a tuple
        config['kernel_size'] = tuple(config['kernel_size'])
        return cls(**config)

class BrainTumorMultiClassCNN:
    def __init__(self, img_height=512, img_width=512, channels=1, num_classes=4):
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.num_classes = num_classes
        self.model = None
        self.evaluator = Evaluator(None, num_classes)

    def build_model(self, hps):
        kernel_size = (5, 5)
        orientations = 8
        psi_init = 0
        num_filters = len(hps['theta'])
        gabor_layer = GaborFilterBank(
            num_filters=num_filters,
            kernel_size=kernel_size,
            orientations=orientations,
            lambd=hps['lambd'],
            sigma=hps['sigma'],
            gamma_init=hps['gamma'],
            psi_init=psi_init,
            theta=hps['theta'],
            bit_widths=hps.get('bit_widths'),
            clusters=hps.get('clusters'),
            trainable=False
        )
        inputs = layers.Input(shape=(self.img_height, self.img_width, self.channels))
        # x = gabor_layer(inputs,hps.get('bit_widths'))
        x = gabor_layer(inputs)
        x = layers.Conv2D(3, kernel_size=1, padding='same')(x)
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_height, self.img_width, 3)
        )
        # Unfreeze the last 50 layers
        for layer in base_model.layers[:-50]:
            layer.trainable = False
        for layer in base_model.layers[-50:]:
            layer.trainable = True
        x = base_model(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hps.get('learning_rate', 0.001) / 10, clipnorm=1.0),  # Lower learning rate for fine-tuning
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        self.model = model
        self.evaluator.model = model
        return model
    
class Evaluator:
    """Handles model evaluation and visualization."""
    def __init__(self, model, num_classes):
        self.model = model
        self.num_classes = num_classes

    def get_predictions_and_labels(self, data, class_indices=None):
        """Get predictions and labels from tf.data.Dataset, handling class subset if needed."""
        if isinstance(data, tf.data.Dataset):
            images, labels = [], []
            for batch_images, batch_labels in data.unbatch().batch(1):
                images.append(batch_images.numpy())
                labels.append(batch_labels.numpy())
            images = np.concatenate(images, axis=0)
            labels = np.concatenate(labels, axis=0)
            predictions = self.model.predict(images)
            y_true = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
            y_true_onehot = labels if labels.ndim > 1 else to_categorical(y_true, num_classes=self.num_classes)
            # If class_indices is provided (e.g., [0, 1, 3]), adjust predictions and labels
            if class_indices is not None:
                # Extract predictions for the relevant classes
                predictions = predictions[:, class_indices]
                # Recompute y_true for the subset of classes
                y_true_mapped = []
                for label in y_true:
                    # Map the original label to the new index in class_indices
                    idx = class_indices.index(label) if label in class_indices else -1
                    y_true_mapped.append(idx)
                y_true_mapped = np.array(y_true_mapped)
                # Filter out samples with unmapped labels (e.g., notumor)
                valid_indices = y_true_mapped != -1
                y_true_mapped = y_true_mapped[valid_indices]
                predictions = predictions[valid_indices]
                y_true_onehot = y_true_onehot[valid_indices][:, class_indices]
            else:
                y_true_mapped = y_true
        else:
            raise ValueError("Expected tf.data.Dataset")
        return predictions, y_true_mapped, y_true_onehot

    def evaluate(self, data, class_indices=None):
        """Evaluate the model using key metrics, handling class subset if needed."""
        predictions, y_true, y_true_onehot = self.get_predictions_and_labels(data, class_indices)
        # Skip evaluation if no valid samples after filtering
        if len(y_true) == 0:
            print("No valid samples to evaluate after filtering classes.")
            return {}
        results = {
            'accuracy': accuracy_score(y_true, np.argmax(predictions, axis=1)),
            'balanced_accuracy': balanced_accuracy_score(y_true, np.argmax(predictions, axis=1)),
            'macro_precision': precision_score(y_true, np.argmax(predictions, axis=1), average='macro'),
            'macro_recall': recall_score(y_true, np.argmax(predictions, axis=1), average='macro'),
            'macro_f1': f1_score(y_true, np.argmax(predictions, axis=1), average='macro'),
            'macro_roc_auc': roc_auc_score(y_true_onehot, predictions, average='macro', multi_class='ovr')
        }
        num_classes = len(class_indices) if class_indices is not None else self.num_classes
        for i in range(num_classes):
            results[f'class_{i}_roc_auc'] = roc_auc_score(y_true_onehot[:, i], predictions[:, i])
        return results

class ModelConfig:
    """Configuration class for model hyperparameters"""
    def __init__(self, **kwargs):
        self.img_height = kwargs.get("img_height", 512)
        self.img_width = kwargs.get("img_width", 512)
        self.channels = kwargs.get("channels", 1)  # Grayscale
        self.num_classes = kwargs.get("num_classes", 4)
        self.batch_size = kwargs.get("batch_size", 16)
        self.epochs = kwargs.get("epochs", 50)

class DataProcessor:
    """Handle data processing and augmentation for MRI images."""
    def __init__(self, config: ModelConfig):
        self.config = config

    def create_kfold_data(self, base_dir: str, n_splits: int = 5, seed: int = 42):
        """Yields tf.data.Dataset objects for training and validation for each fold."""
        df, self.class_indices = self._load_image_paths_and_labels(base_dir)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        class_names = list(self.class_indices.keys())
        num_classes = len(class_names)

        def augment_image(image, label):
            image = tf.image.random_flip_left_right(image, seed=seed)
            angle = tf.random.uniform([], -15 * np.pi / 180, 15 * np.pi / 180, dtype=tf.float32)
            image = tf.image.rot90(image, k=tf.cast(angle * 4 / (2 * np.pi), tf.int32))
            image = tf.image.random_brightness(image, max_delta=0.1, seed=seed)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1, seed=seed)
            scale = tf.random.uniform([], 0.9, 1.1, dtype=tf.float32)
            new_height = tf.cast(tf.cast(self.config.img_height, tf.float32) * scale, tf.int32)
            new_width = tf.cast(tf.cast(self.config.img_width, tf.float32) * scale, tf.int32)
            image = tf.image.resize(image, [new_height, new_width])
            image = tf.image.resize_with_crop_or_pad(image, self.config.img_height, self.config.img_width)
            # Shear transformation
            shear = tf.random.uniform([], -0.1, 0.1, dtype=tf.float32)
            # Construct a 3x3 projective transform matrix for shear
            shear_matrix = tf.stack([
                tf.constant(1.0), shear, tf.constant(0.0),  # [1, s, 0]
                tf.constant(0.0), tf.constant(1.0), tf.constant(0.0),  # [0, 1, 0]
                tf.constant(0.0), tf.constant(0.0), tf.constant(1.0)   # [0, 0, 1]
            ])
            # Flatten to [9], then take first 8 elements for projective transform
            shear_matrix = tf.reshape(shear_matrix, [9])[:8]  # [1, s, 0, 0, 1, 0, 0, 0]
            shear_matrix = tf.expand_dims(shear_matrix, 0)  # Shape: [1, 8]
            image = tf.raw_ops.ImageProjectiveTransformV3(
                images=tf.expand_dims(image, 0),
                transforms=shear_matrix,
                output_shape=[self.config.img_height, self.config.img_width],
                fill_value=0.0,
                interpolation='BILINEAR'
            )[0]
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02, dtype=tf.float32)
            image = image + noise
            shift_fraction = 0.05
            crop_height = int(self.config.img_height * (1 - 2 * shift_fraction))
            crop_width = int(self.config.img_width * (1 - 2 * shift_fraction))
            image = tf.image.random_crop(
                image,
                size=[crop_height, crop_width, self.config.channels],
                seed=seed
            )
            max_offset_height = int(self.config.img_height * shift_fraction)
            max_offset_width = int(self.config.img_width * shift_fraction)
            offset_height = tf.random.uniform([], 0, max_offset_height, dtype=tf.int32)
            offset_width = tf.random.uniform([], 0, max_offset_width, dtype=tf.int32)
            image = tf.image.pad_to_bounding_box(
                image,
                offset_height=offset_height,
                offset_width=offset_width,
                target_height=self.config.img_height,
                target_width=self.config.img_width
            )
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label

        def load_and_preprocess(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=self.config.channels)
            image = tf.image.resize(image, [self.config.img_height, self.config.img_width])
            image = tf.py_function(self._normalize_image, [image], tf.float32)
            image.set_shape([self.config.img_height, self.config.img_width, self.config.channels])
            return image, tf.one_hot(label, num_classes)

        for fold, (train_idx, val_idx) in enumerate(skf.split(df.filepath, df.label)):
            print(f"\n📂 Fold {fold + 1}/{n_splits}")
            train_paths = df.filepath.iloc[train_idx].values
            train_labels = df.label.iloc[train_idx].values
            val_paths = df.filepath.iloc[val_idx].values
            val_labels = df.label.iloc[val_idx].values
            train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
            train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
            train_ds = train_ds.shuffle(1024).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
            val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
            val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
            yield fold, train_ds, val_ds

    def create_full_dataset(self, train_dir: str, validation_dir: str):
        """Create tf.data.Dataset for full training and validation sets."""
        train_df, class_indices = self._load_image_paths_and_labels(train_dir)
        val_df, _ = self._load_image_paths_and_labels(validation_dir)
        num_classes = len(class_indices)

        def augment_image(image, label):
            image = tf.image.random_flip_left_right(image, seed=42)
            angle = tf.random.uniform([], -15 * np.pi / 180, 15 * np.pi / 180, dtype=tf.float32)
            image = tf.image.rot90(image, k=tf.cast(angle * 4 / (2 * np.pi), tf.int32))
            image = tf.image.random_brightness(image, max_delta=0.1, seed=42)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1, seed=42)
            scale = tf.random.uniform([], 0.9, 1.1, dtype=tf.float32)
            new_height = tf.cast(tf.cast(self.config.img_height, tf.float32) * scale, tf.int32)
            new_width = tf.cast(tf.cast(self.config.img_width, tf.float32) * scale, tf.int32)
            image = tf.image.resize(image, [new_height, new_width])
            image = tf.image.resize_with_crop_or_pad(image, self.config.img_height, self.config.img_width)
            shear = tf.random.uniform([], -0.1, 0.1, dtype=tf.float32)
            shear_matrix = tf.stack([
                tf.constant(1.0), shear, tf.constant(0.0),
                tf.constant(0.0), tf.constant(1.0), tf.constant(0.0),
                tf.constant(0.0), tf.constant(0.0), tf.constant(1.0)
            ])
            shear_matrix = tf.reshape(shear_matrix, [9])[:8]
            shear_matrix = tf.expand_dims(shear_matrix, 0)
            image = tf.raw_ops.ImageProjectiveTransformV3(
                images=tf.expand_dims(image, 0),
                transforms=shear_matrix,
                output_shape=[self.config.img_height, self.config.img_width],
                fill_value=0.0,
                interpolation='BILINEAR'
            )[0]
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02, dtype=tf.float32)
            image = image + noise
            shift_fraction = 0.05
            crop_height = int(self.config.img_height * (1 - 2 * shift_fraction))
            crop_width = int(self.config.img_width * (1 - 2 * shift_fraction))
            image = tf.image.random_crop(
                image,
                size=[crop_height, crop_width, self.config.channels],
                seed=42
            )
            max_offset_height = int(self.config.img_height * shift_fraction)
            max_offset_width = int(self.config.img_width * shift_fraction)
            offset_height = tf.random.uniform([], 0, max_offset_height, dtype=tf.int32)
            offset_width = tf.random.uniform([], 0, max_offset_width, dtype=tf.int32)
            image = tf.image.pad_to_bounding_box(
                image,
                offset_height=offset_height,
                offset_width=offset_width,
                target_height=self.config.img_height,
                target_width=self.config.img_width
            )
            image = tf.clip_by_value(image, 0.0, 1.0)
            return image, label

        def load_and_preprocess(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=self.config.channels)
            image = tf.image.resize(image, [self.config.img_height, self.config.img_width])
            image = tf.py_function(self._normalize_image, [image], tf.float32)
            image.set_shape([self.config.img_height, self.config.img_width, self.config.channels])
            return image, tf.one_hot(label, num_classes)

        train_ds = tf.data.Dataset.from_tensor_slices((train_df.filepath, train_df.label))
        train_ds = train_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.shuffle(1024).batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.data.Dataset.from_tensor_slices((val_df.filepath, val_df.label))
        val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        return train_ds, val_ds

    def create_hardware_validation_dataset(self, base_dir: str):
        """Create tf.data.Dataset for validation on hardware Gabor-filtered dataset."""
        data = []
        class_names = sorted(os.listdir(base_dir))
        label_map = {'glioma': 0, 'meningioma': 1, 'notumor': 2}
        for class_name in class_names:
            class_path = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for run_name in os.listdir(class_path):
                run_path = os.path.join(class_path, run_name)
                if not os.path.isdir(run_path):
                    continue
                run_images = sorted([f for f in os.listdir(run_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if len(run_images) != 4:
                    print(f"Warning: Run {run_path} has {len(run_images)} images, expected 4. Skipping.")
                    continue
                run_image_paths = [os.path.join(run_path, img) for img in run_images]
                data.append((run_image_paths, class_name))
        
        if not data:
            raise ValueError("No valid runs found in hardware dataset directory")

        df = pd.DataFrame(data, columns=["image_paths", "class_name"])
        df["label"] = df.class_name.map(label_map)
        
        def load_and_stack_images(image_paths, label):
            images = []
            for path in image_paths:
                image = tf.io.read_file(path)
                image = tf.image.decode_jpeg(image, channels=1)
                image = tf.image.resize(image, [self.config.img_height, self.config.img_width])
                image = tf.py_function(self._normalize_image, [image], tf.float32)
                image.set_shape([self.config.img_height, self.config.img_width, 1])
                images.append(image)
            stacked_image = tf.concat(images, axis=-1)
            return stacked_image, tf.one_hot(label, depth=4)

        dataset = tf.data.Dataset.from_tensor_slices((df.image_paths, df.label))
        dataset = dataset.map(load_and_stack_images, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset, class_names

    def compute_class_weights_from_dataset(self, dataset):
        """Calculate class weights to handle imbalance from tf.data.Dataset."""
        labels = []
        for _, label in dataset.unbatch():
            labels.append(np.argmax(label.numpy()))
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        class_weights = {}
        for cls, count in enumerate(class_counts):
            weight = total_samples / (len(class_counts) * count)
            if cls == 0:
                weight *= 2.0
            class_weights[cls] = weight
        return class_weights

    def _load_image_paths_and_labels(self, base_dir: str):
        """Return a DataFrame with filepaths and class labels."""
        data = []
        for class_name in sorted(os.listdir(base_dir)):
            class_path = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append((os.path.join(class_path, fname), class_name))
        df = pd.DataFrame(data, columns=["filepath", "class_name"])
        label_map = {name: idx for idx, name in enumerate(sorted(df.class_name.unique()))}
        df["label"] = df.class_name.map(label_map)
        return df, label_map

    @staticmethod
    def _normalize_image(x: np.ndarray) -> np.ndarray:
        """Normalize image using z-score normalization."""
        return (x - np.mean(x)) / (np.maximum(np.std(x), 1e-7))

def run_optimization(brain_tumor_model, hps, train_ds, val_ds, class_weights):
    with tf.device("/GPU:0"):
        tf.keras.backend.clear_session()
        brain_tumor_model.build_model(hps)
        brain_tumor_model.model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0,class_weight=class_weights)
    brain_tumor_model.evaluator.model = brain_tumor_model.model
    metrics = brain_tumor_model.evaluator.evaluate(val_ds)
    with open("fold_metric_optuna.txt","w") as f:
        f.write(f"{metrics['macro_f1']}")
        f.flush()
    with tf.device('/CPU:0'):
        del brain_tumor_model.model
        del brain_tumor_model.evaluator.model
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Brain Tumor Model")
    
    # Add arguments for each required parameter
    parser.add_argument("hps")
    parser.add_argument("training_path")
    parser.add_argument("testing_path")
    parser.add_argument("class_weights")
    
    args = parser.parse_args()

    # Deserialize best_hps from JSON string

    print("hello",flush=True)

    # # You will need to load the model and datasets appropriately
    brain_tumor_model = BrainTumorMultiClassCNN(512,512,1,4)  # Replace with actual model loading

    class_weights = eval(args.class_weights)  # Assuming it's a dictionary, you may need to adjust this
    hps = eval(args.hps)

    # print(hps.get('bit_widths'))

    train_dir, validation_dir = args.training_path, args.testing_path

    config = ModelConfig(img_height=512, img_width=512, channels=1, batch_size=16)

    if not all(map(os.path.exists, [train_dir, validation_dir])):
        raise ValueError("Training or validation directory missing")
    
    data_processor = DataProcessor(config)
    train_ds, val_ds = data_processor.create_full_dataset(train_dir, validation_dir)

    # # Run the training with the deserialized `best_hps`
    run_optimization(brain_tumor_model, hps, train_ds, val_ds, class_weights)
