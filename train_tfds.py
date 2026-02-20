import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19

def preprocess(image, label):
    # Resize to expected VGG19 input size
    image = tf.image.resize(image, (224, 224))
    # VGG19 preprocess_input expects 0-255 in BGR usually, but tf.keras.applications handles it.
    # However, to be safe and standard with VGG19 from keras applications:
    # We should NOT normalize to [0,1] manually if using preprocess_input on raw images.
    # But tfds images are [0,255].
    # Let's use the explicit preprocess_input function.
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image, label

def build_and_train(output_path='models/dogbreed.h5', epochs=1, batch_size=32):
    print("Loading Oxford-IIIT Pet dataset...")
    # split='train[:20%]' for speed if needed, but let's try full train first.
    # Actually, for a quick "run" to fix the app, maybe just use a subset to be fast?
    # The user said "run", implying they want it working. A weak model is better than no model.
    # I'll use 40% of data to be faster. 
    # Wait, the user might want a good model. I'll use 50% for now.
    (ds_train, ds_test), ds_info = tfds.load(
        'oxford_iiit_pet',
        split=['train[:50%]', 'test[:10%]'], # Small subset for speed demonstration
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    num_classes = ds_info.features['label'].num_classes
    # labels are 0-36.
    class_names = ds_info.features['label'].names
    
    print(f"Classes: {len(class_names)}")

    # Preprocess
    ds_train = ds_train.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build Model
    print("Building VGG19 model...")
    # include_top=False means we get the conv features.
    base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    # We already preprocessed in the dataset map, so we just pass through.
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Training...")
    model.fit(ds_train, validation_data=ds_test, epochs=epochs)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    
    label_path = os.path.join(os.path.dirname(output_path), 'labels.txt')
    with open(label_path, 'w') as f:
        f.write('\n'.join(class_names))

    print(f'Saved model to {output_path} and labels to {label_path}')

if __name__ == '__main__':
    build_and_train()
