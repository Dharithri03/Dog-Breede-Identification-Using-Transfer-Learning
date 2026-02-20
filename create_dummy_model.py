
import os
import tensorflow as tf
from tensorflow.keras import layers, models

def create_dummy_model(output_path='models/dogbreed.h5'):
    print("Creating dummy model...")
    # Create a simple valid model that accepts 224x224x3 input
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax') # Assumes 10 classes for dummy
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"Saved dummy model to {output_path}")
    
    # Create dummy labels
    labels = [f'Dummy_Breed_{i}' for i in range(10)]
    label_path = os.path.join(os.path.dirname(output_path), 'labels.txt')
    with open(label_path, 'w') as f:
        f.write('\n'.join(labels))
    print(f"Saved dummy labels to {label_path}")

if __name__ == '__main__':
    create_dummy_model()
