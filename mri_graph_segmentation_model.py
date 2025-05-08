"""
MRI Brain Segmentation Model using TensorFlow
============================================
This module implements a deep learning model for MRI brain segmentation
which is then used to build graphs for structural analysis.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

class MRISegmentationModel:
    def __init__(self, input_shape=(256, 256, 1), model_dir="models"):
        """
        Initialize the MRI segmentation model
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input images (height, width, channels)
        model_dir : str
            Directory to save model checkpoints
        """
        self.input_shape = input_shape
        self.model_dir = model_dir
        self.model = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def build_unet_model(self, filters_base=64):
        """
        Build a U-Net model for semantic segmentation
        
        Parameters:
        -----------
        filters_base : int
            Base number of filters (doubled in each downsampling step)
            
        Returns:
        --------
        tensorflow.keras.Model
            Compiled U-Net model
        """
        inputs = layers.Input(self.input_shape)
        
        # Contracting path (encoder)
        # Level 1
        conv1 = layers.Conv2D(filters_base, 3, activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(filters_base, 3, activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        
        # Level 2
        conv2 = layers.Conv2D(filters_base*2, 3, activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(filters_base*2, 3, activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        
        # Level 3
        conv3 = layers.Conv2D(filters_base*4, 3, activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(filters_base*4, 3, activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Level 4
        conv4 = layers.Conv2D(filters_base*8, 3, activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(filters_base*8, 3, activation='relu', padding='same')(conv4)
        drop4 = layers.Dropout(0.5)(conv4)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)
        
        # Bottom level
        conv5 = layers.Conv2D(filters_base*16, 3, activation='relu', padding='same')(pool4)
        conv5 = layers.Conv2D(filters_base*16, 3, activation='relu', padding='same')(conv5)
        drop5 = layers.Dropout(0.5)(conv5)
        
        # Expansive path (decoder)
        # Level 4
        up6 = layers.Conv2DTranspose(filters_base*8, 2, strides=(2, 2), padding='same')(drop5)
        merge6 = layers.concatenate([drop4, up6], axis=3)
        conv6 = layers.Conv2D(filters_base*8, 3, activation='relu', padding='same')(merge6)
        conv6 = layers.Conv2D(filters_base*8, 3, activation='relu', padding='same')(conv6)
        
        # Level 3
        up7 = layers.Conv2DTranspose(filters_base*4, 2, strides=(2, 2), padding='same')(conv6)
        merge7 = layers.concatenate([conv3, up7], axis=3)
        conv7 = layers.Conv2D(filters_base*4, 3, activation='relu', padding='same')(merge7)
        conv7 = layers.Conv2D(filters_base*4, 3, activation='relu', padding='same')(conv7)
        
        # Level 2
        up8 = layers.Conv2DTranspose(filters_base*2, 2, strides=(2, 2), padding='same')(conv7)
        merge8 = layers.concatenate([conv2, up8], axis=3)
        conv8 = layers.Conv2D(filters_base*2, 3, activation='relu', padding='same')(merge8)
        conv8 = layers.Conv2D(filters_base*2, 3, activation='relu', padding='same')(conv8)
        
        # Level 1
        up9 = layers.Conv2DTranspose(filters_base, 2, strides=(2, 2), padding='same')(conv8)
        merge9 = layers.concatenate([conv1, up9], axis=3)
        conv9 = layers.Conv2D(filters_base, 3, activation='relu', padding='same')(merge9)
        conv9 = layers.Conv2D(filters_base, 3, activation='relu', padding='same')(conv9)
        
        # Output layer - we use sigmoid for binary segmentation
        # For multi-class segmentation, change to softmax with appropriate number of classes
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model with appropriate loss and metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])]
        )
        
        self.model = model
        return model
    
    def data_generator(self, images, masks, batch_size=8, augment=True):
        """
        Generator to yield batches of image/mask pairs with optional augmentation
        
        Parameters:
        -----------
        images : ndarray
            Input MRI images
        masks : ndarray
            Ground truth segmentation masks
        batch_size : int
            Batch size
        augment : bool
            Whether to apply data augmentation
            
        Yields:
        -------
        tuple
            Batch of (images, masks)
        """
        num_samples = len(images)
        indices = np.arange(num_samples)
        
        while True:
            np.random.shuffle(indices)
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_images = images[batch_indices]
                batch_masks = masks[batch_indices]
                
                if augment:
                    # Apply augmentations
                    for j in range(len(batch_images)):
                        if np.random.rand() < 0.5:
                            # Horizontal flip
                            batch_images[j] = np.fliplr(batch_images[j])
                            batch_masks[j] = np.fliplr(batch_masks[j])
                        
                        if np.random.rand() < 0.5:
                            # Vertical flip
                            batch_images[j] = np.flipud(batch_images[j])
                            batch_masks[j] = np.flipud(batch_masks[j])
                        
                        if np.random.rand() < 0.3:
                            # Random rotation (90, 180, or 270 degrees)
                            k = np.random.randint(1, 4)
                            batch_images[j] = np.rot90(batch_images[j], k=k)
                            batch_masks[j] = np.rot90(batch_masks[j], k=k)
                            
                        if np.random.rand() < 0.3:
                            # Random brightness adjustment
                            brightness_factor = np.random.uniform(0.8, 1.2)
                            batch_images[j] = np.clip(batch_images[j] * brightness_factor, 0, 1)
                
                # Reshape for model input if needed
                if len(batch_images.shape) == 3:
                    batch_images = batch_images[..., np.newaxis]
                if len(batch_masks.shape) == 3:
                    batch_masks = batch_masks[..., np.newaxis]
                
                yield batch_images, batch_masks
    
    def prepare_synthetic_dataset(self, num_samples=200, size=(256, 256)):
        """
        Create a synthetic dataset of MRI images and segmentation masks
        
        Parameters:
        -----------
        num_samples : int
            Number of samples to generate
        size : tuple
            Size of the images (height, width)
            
        Returns:
        --------
        tuple
            (images, masks) arrays
        """
        images = []
        masks = []
        
        for _ in range(num_samples):
            # Create empty image and mask
            image = np.zeros(size)
            mask = np.zeros(size)
            
            # Create brain outline (ellipse)
            center_x = size[0] // 2 + np.random.randint(-20, 20)
            center_y = size[1] // 2 + np.random.randint(-20, 20)
            a = np.random.randint(70, 100)  # semi-major axis
            b = np.random.randint(90, 120)  # semi-minor axis
            
            y, x = np.ogrid[-center_x:size[0]-center_x, -center_y:size[1]-center_y]
            brain_mask = (x*x)/(a*a) + (y*y)/(b*b) <= 1
            
            # Add brain tissue
            image[brain_mask] = np.random.uniform(0.5, 0.8)
            mask[brain_mask] = 1
            
            # Add regions inside the brain
            num_regions = np.random.randint(3, 8)
            
            for _ in range(num_regions):
                # Random position inside the brain
                while True:
                    pos_x = np.random.randint(0, size[0])
                    pos_y = np.random.randint(0, size[1])
                    if brain_mask[pos_x, pos_y]:
                        break
                
                radius = np.random.randint(10, 30)
                
                # Create region mask
                y, x = np.ogrid[-pos_x:size[0]-pos_x, -pos_y:size[1]-pos_y]
                region_mask = x*x + y*y <= radius*radius
                
                # Ensure region is inside brain
                region_mask = np.logical_and(region_mask, brain_mask)
                
                # Add region to image with higher intensity
                image[region_mask] = np.random.uniform(0.8, 1.0)
            
            # Add noise
            noise = np.random.normal(0, 0.05, size)
            image = image + noise
            image = np.clip(image, 0, 1)
            
            images.append(image)
            masks.append(mask)
        
        return np.array(images), np.array(masks)
    
    def train(self, images, masks, batch_size=8, epochs=50, validation_split=0.2):
        """
        Train the segmentation model
        
        Parameters:
        -----------
        images : ndarray
            Training images
        masks : ndarray
            Ground truth masks
        batch_size : int
            Batch size for training
        epochs : int
            Number of epochs
        validation_split : float
            Fraction of data to use for validation
            
        Returns:
        --------
        History object
            Training history
        """
        if self.model is None:
            self.build_unet_model()
        
        # Split data
        train_images, val_images, train_masks, val_masks = train_test_split(
            images, masks, test_size=validation_split, random_state=42)
        
        # Add channel dimension if needed
        if len(train_images.shape) == 3:
            train_images = train_images[..., np.newaxis]
            val_images = val_images[..., np.newaxis]
            
        if len(train_masks.shape) == 3:
            train_masks = train_masks[..., np.newaxis]
            val_masks = val_masks[..., np.newaxis]
        
        # Prepare data generators
        train_gen = self.data_generator(train_images, train_masks, batch_size, augment=True)
        val_gen = self.data_generator(val_images, val_masks, batch_size, augment=False)
        
        # Prepare callbacks
        checkpoint = ModelCheckpoint(
            os.path.join(self.model_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        callbacks = [checkpoint, early_stopping, reduce_lr]
        
        # Train model
        steps_per_epoch = len(train_images) // batch_size
        validation_steps = len(val_images) // batch_size
        
        history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        
        return history
    
    def load_model(self, model_path):
        """Load a pretrained model"""
        self.model = tf.keras.models.load_model(model_path)
        return self.model
    
    def predict(self, image):
        """
        Predict segmentation mask for a single image
        
        Parameters:
        -----------
        image : ndarray
            Input MRI image
            
        Returns:
        --------
        ndarray
            Predicted segmentation mask
        """
        if self.model is None:
            raise ValueError("Model not built or loaded yet")
        
        # Add batch and channel dimensions if needed
        if len(image.shape) == 2:
            image = image[np.newaxis, ..., np.newaxis]
        elif len(image.shape) == 3 and image.shape[0] != 1:
            image = image[np.newaxis, ...]
        
        # Predict mask
        predicted_mask = self.model.predict(image)[0]
        
        # Threshold to get binary mask
        binary_mask = (predicted_mask > 0.5).astype(np.uint8)
        
        return binary_mask
    
    def visualize_predictions(self, images, true_masks=None, save_dir=None):
        """
        Visualize model predictions against ground truth
        
        Parameters:
        -----------
        images : ndarray
            Input images
        true_masks : ndarray, optional
            Ground truth masks
        save_dir : str, optional
            Directory to save visualizations
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for i, image in enumerate(images):
            # Get prediction
            pred_mask = self.predict(image)
            
            # Plot results
            plt.figure(figsize=(12, 4))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Original MRI')
            plt.axis('off')
            
            # Predicted mask
            plt.subplot(1, 3, 2)
            plt.imshow(pred_mask[..., 0], cmap='viridis')
            plt.title('Predicted Mask')
            plt.axis('off')
            
            # True mask (if available)
            if true_masks is not None:
                plt.subplot(1, 3, 3)
                if len(true_masks.shape) == 4:
                    plt.imshow(true_masks[i, ..., 0], cmap='viridis')
                else:
                    plt.imshow(true_masks[i], cmap='viridis')
                plt.title('True Mask')
                plt.axis('off')
            
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'prediction_{i}.png'), dpi=200)
                plt.close()
            else:
                plt.show()