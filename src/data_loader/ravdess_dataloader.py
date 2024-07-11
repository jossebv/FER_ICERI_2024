import pandas as pd
import cv2
from src.landmarks_utils import face_get_XYZ
import mediapipe as mp
import tensorflow as tf


class RavdessDataloader:
    def __init__(self, annot_path, split, random_seed):
        self.annot_path = annot_path
        self.split = split
        self.random_seed = random_seed

        self.annotations = pd.read_csv(annot_path).sample(
            frac=1, random_state=random_seed
        )
        self.len = len(self.annotations)
        self.mp_detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def generator(self):
        idx = 0
        while idx < self.len:
            frame_annotation = self.annotations.iloc[idx]
            frame_path = frame_annotation[0]
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_detector.process(frame)
            _, landmarks = face_get_XYZ(results, frame)
            emotion_id = frame_annotation[1]
            yield landmarks, emotion_id
            idx += 1


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

    dataloader = dataloader.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataloader
