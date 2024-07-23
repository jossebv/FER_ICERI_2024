import pandas as pd
import cv2
import sys
import os
import numpy as np
import tensorflow as tf


class FER_2013Dataloader:
    def __init__(
        self,
        data_path,
        split,
        random_seed,
    ):

        print("[Preparing dataloader]")

        annot_path = os.path.join(data_path, f"{split}_landmarks_annotations.csv")
        self.split = split
        print(f"\tSplit: {split}")
        self.random_seed = random_seed

        self.data = pd.read_csv(annot_path).sample(frac=1, random_state=random_seed)
        self.data_len = len(self.data)

    def label_to_id(self, label):
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        return emotions.index(label)

    def generator(self):
        idx = 0
        while idx < self.data_len:
            frame_annotation = self.data.iloc[idx]
            landmarks_path = frame_annotation.loc["path"]
            landmarks = np.load(landmarks_path)
            emotion_label = frame_annotation.loc["emotion"]
            emotion_id = self.label_to_id(emotion_label)
            yield landmarks, emotion_id
            idx += 1


def preprocess(landmarks, label):
    landmarks = tf.image.convert_image_dtype(
        landmarks[..., tf.newaxis], dtype=tf.float32
    )

    label = tf.cast(label, tf.float32)
    return landmarks, label


def prepare_dataloader(data_path, split, batch_size, random_seed=42):
    dataset = FER_2013Dataloader(data_path, split, random_seed)
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
    DATA_PATH = "data/FER-2013"
    trainloader = prepare_dataloader(
        DATA_PATH,
        "train",
        4,
    )
    testloader = prepare_dataloader(DATA_PATH, "test", 4)

    print(next(iter(trainloader)))
    print(next(iter(testloader)))

    print("Dataloader works correctly")
