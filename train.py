"""
Simple training script (expects directory structure:
  data/train/<class>/*.jpg
  data/validation/<class>/*.jpg

This script demonstrates building a classifier using VGG19 as backbone and saving `models/dogbreed.h5` and `models/labels.txt`.
"""
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19


def build_and_train(data_dir, output_path='models/dogbreed.h5', img_size=(224,224), batch_size=32, epochs=5):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'validation')

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch_size)

    class_names = train_ds.class_names

    base = VGG19(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    base.trainable = False

    inputs = layers.Input(shape=(*img_size, 3))
    x = tf.keras.applications.vgg19.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    with open(os.path.join(os.path.dirname(output_path), 'labels.txt'), 'w') as f:
        f.write('\n'.join(class_names))

    print('Saved model to', output_path)


if __name__ == '__main__':
    # Example usage: ensure you prepared data/ with train/ & validation/ folders
    build_and_train('data', output_path='models/dogbreed.h5')
