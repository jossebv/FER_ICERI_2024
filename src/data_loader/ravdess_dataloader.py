import pandas as pd
import cv2
import sys
import os
import numpy as np

sys.path.append(".")
from src.landmarks_utils import face_get_XYZ
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split


class RavdessDataloader:
    def __init__(
        self,
        annot_path,
        split,
        random_seed,
    ):
        self.annot_path = annot_path
        self.split = split
        self.random_seed = random_seed

        self.annotations = pd.read_csv(annot_path).sample(
            frac=1, random_state=random_seed
        )
        train, test = train_test_split(self.annotations, random_state=random_seed)

        print(
            f"Total len: {len(self.annotations)}:\n\t- Train: {len(train)}\n\t- Test: {len(test)}"
        )

        self.data = train if self.split == "train" else test
        self.data_len = len(self.data)

    def generator(self):
        idx = 0
        while idx < self.data_len:
            frame_annotation = self.data.iloc[idx]
            frame_path = frame_annotation[0]
            landmarks_path = frame_path.replace("frames", "landmarks").replace(
                "jpg", "npy"
            )
            landmarks = np.load(landmarks_path)
            emotion_id = frame_annotation[1] - 1  # Emotions in range [0,7]
            yield landmarks, emotion_id
            idx += 1


def preprocess(landmarks, label):
    landmarks = tf.image.convert_image_dtype(
        landmarks[..., tf.newaxis], dtype=tf.float32
    )

    label = tf.cast(label, tf.float32)
    return landmarks, label


def prepare_dataloader(annotations_path, split, batch_size, random_seed=42):
    dataset = RavdessDataloader(annotations_path, split, random_seed)
    dataloader = tf.data.Dataset.from_generator(
        generator=dataset.generator,
        output_signature=(
            tf.TensorSpec(
                shape=(478, 2), dtype=tf.float32
            ),  # Shape and type of the landmarks
            tf.TensorSpec(shape=(), dtype=tf.float32),  # Shape and type of the labels
        ),
    )

    dataloader = (
        dataloader.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataloader


if __name__ == "__main__":
    ANNOTATION_PATH = "data/RAVDESS/annotations_frames.csv"
    trainloader = prepare_dataloader(
        ANNOTATION_PATH,
        "train",
        4,
    )

    print(next(iter(trainloader)))
