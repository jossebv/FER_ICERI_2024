import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

sys.path.append(".")
from src.landmarks_utils import face_get_XYZ, normalize_L0, normalize_size

# RESOLUTIONS
HIGHRES_SIZE = (1280, 720)
LARGE_SIZE = (640, 480)
SMALL_SIZE = (320, 200)

DATA_DIR = "/mnt/RESOURCES/josemanuelbravo/FER_2024/data/FER-2013"
IMAGES_DIR = os.path.join(DATA_DIR, "raw")
LANDMARKS_DIR = os.path.join(DATA_DIR, "landmarks")

emotions = os.listdir(os.path.join(IMAGES_DIR, "train"))


def create_dir_tree():
    for subset in ["train", "test"]:
        for emotion in emotions:
            path = os.path.join(LANDMARKS_DIR, subset, emotion)
            os.makedirs(path, exist_ok=True)


def extract_landmarks(frame_path, mp_detector):
    frame = cv2.imread(frame_path)
    frame = cv2.resize(frame, HIGHRES_SIZE)
    results = mp_detector.process(frame)
    frame, landmarks = face_get_XYZ(results, frame)

    return landmarks


def save_landmarks(save_path, landmarks):
    np.save(save_path, landmarks)


def main():
    mp_detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    create_dir_tree()

    for subset in ["train", "test"]:
        landmarks_subset_path = os.path.join(LANDMARKS_DIR, subset)
        annotations = pd.read_csv(os.path.join(DATA_DIR, f"{subset}_annotations.csv"))
        landmarks_annotations = pd.DataFrame([], columns=["path", "emotion"])

        for _, image in tqdm(
            annotations.iterrows(), desc="Extracting landmarks", total=len(annotations)
        ):
            image_path = image.loc["path"]
            image_class = image.loc["emotion"]
            landmarks = extract_landmarks(image_path, mp_detector)

            if landmarks.sum() == 0:
                continue

            landmarks = normalize_L0(landmarks)
            landmarks = normalize_size(landmarks)

            save_path = image_path.replace("raw", "landmarks").replace(".jpg", ".npy")
            image_annot = pd.Series({"path": save_path, "emotion": image_class})
            landmarks_annotations = pd.concat(
                (landmarks_annotations, image_annot.to_frame().T), ignore_index=True
            )
            save_landmarks(save_path, landmarks)

        landmarks_annotations.to_csv(
            os.path.join(DATA_DIR, f"{subset}_landmarks_annotations.csv")
        )


if __name__ == "__main__":
    main()
