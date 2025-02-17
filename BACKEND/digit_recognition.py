import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

class DigitRecognizer:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.model = None
        self.best_model_path = os.path.join(models_dir, 'best_model.h5')
        self.metrics_path = os.path.join(models_dir, 'model_metrics.json')
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
    def create_model_basic(self):
        """Basic CNN architecture"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax')
        ])
        return model
    
    def create_model_deep(self):
        """Deeper CNN architecture"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(64, (3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        return model
    
    def create_model_residual(self):
        """Residual CNN architecture"""
        inputs = layers.Input(shape=(28, 28, 1))
        
        # First conv block
        x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Residual connection
        shortcut = layers.Conv2D(32, (1, 1), padding='same')(inputs)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Second conv block with residual
        prev = x
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        shortcut = layers.Conv2D(64, (1, 1), padding='same')(prev)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        
        return models.Model(inputs=inputs, outputs=outputs)
    
    def load_and_preprocess_data(self):
        """Load and preprocess MNIST dataset"""
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        
        # Reshape and normalize
        train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
        test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
        
        return (train_images, train_labels), (test_images, test_labels)
    
    def train_and_evaluate_model(self, model, name, train_data, test_data):
        """Train and evaluate a single model"""
        train_images, train_labels = train_data
        test_images, test_labels = test_data
        
        # Compile model
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Data augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1
        )
        
        # Train
        print(f"\nTraining {name}...")
        history = model.fit(
            datagen.flow(train_images, train_labels, batch_size=128),
            epochs=15,
            validation_data=(test_images, test_labels),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        return history, test_acc
    
    def train_all_models(self):
        """Train and compare all model architectures"""
        # Load data
        train_data, test_data = self.load_and_preprocess_data()
        
        # Define models to try
        models_to_try = {
            'basic': self.create_model_basic(),
            'deep': self.create_model_deep(),
            'residual': self.create_model_residual()
        }
        
        # Train and evaluate each model
        results = {}
        best_acc = 0
        best_model = None
        
        for name, model in models_to_try.items():
            history, accuracy = self.train_and_evaluate_model(
                model, name, train_data, test_data)
            results[name] = {
                'accuracy': float(accuracy),
                'history': {
                    'accuracy': [float(x) for x in history.history['accuracy']],
                    'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                    'loss': [float(x) for x in history.history['loss']],
                    'val_loss': [float(x) for x in history.history['val_loss']]
                }
            }
            
            if accuracy > best_acc:
                best_acc = accuracy
                best_model = model
        
        # Save results
        with open(self.metrics_path, 'w') as f:
            json.dump(results, f)
        
        # Save best model
        if best_model is not None:
            best_model.save(self.best_model_path)
            self.model = best_model
        
        return results
    
    def load_best_model(self):
        """Load the best model if it exists"""
        if os.path.exists(self.best_model_path):
            self.model = models.load_model(self.best_model_path)
            return True
        return False
    
    def predict(self, image):
        """Predict digit from image"""
        if self.model is None:
            if not self.load_best_model():
                raise Exception("No model available. Please train models first.")
        
        # Ensure image is in correct shape
        if len(image.shape) == 2:
            image = image.reshape(1, 28, 28, 1)
        elif len(image.shape) == 3:
            image = image.reshape(1, *image.shape)
        
        prediction = self.model.predict(image, verbose=0)
        return np.argmax(prediction), np.max(prediction)
    
    def plot_training_results(self):
        """Plot training results for all models"""
        if not os.path.exists(self.metrics_path):
            print("No training results available")
            return
        
        with open(self.metrics_path, 'r') as f:
            results = json.load(f)
        
        plt.figure(figsize=(15, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        for model_name, metrics in results.items():
            plt.plot(metrics['history']['val_accuracy'], 
                    label=f"{model_name} (max: {metrics['accuracy']:.4f})")
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        for model_name, metrics in results.items():
            plt.plot(metrics['history']['val_loss'], label=model_name)
        plt.title('Model Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()