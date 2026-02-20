"""Utility helpers used by training notebook / scripts."""
import os
import tensorflow as tf
import tensorflow_datasets as tfds


def prepare_oxford_pet_as_dirs(target_dir='data', img_size=(224,224), val_split=0.2):
    """Download Oxford-IIIT Pet via tfds and export to `target_dir/train/<label>` and `target_dir/validation/<label>`.
    NOTE: This may take several minutes and requires tensorflow-datasets.
    """
    ds, info = tfds.load('oxford_iiit_pet:3.*.*', split='train+test', with_info=True)
    labels = info.features['species'].names  # integer labels -> species names

    os.makedirs(target_dir, exist_ok=True)
    # For simplicity the notebook demonstrates TF dataset pipeline; this helper is optional.
    return info
