import keras
import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from cameras import CVCamera
from landmarks_utils import (
    face_get_XYZ,
    normalize_L0,
    normalize_size,
)

# RESOLUTIONS
HIGHRES_SIZE = (1280, 720)
LARGE_SIZE = (640, 480)
SMALL_SIZE = (320, 200)

# COLORS
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

# FRAME RATE
FPS = 30

# classes = [
#     "neutral",
#     "calm",
#     "happy",
#     "sad",
#     "angry",
#     "fearful",
#     "disgust",
#     "surprised",
# ]

my_recorded_classes = ["Angry", "Happy", "Sad", "Surprise"]


def get_face_corners(landmarks, image_size):
    maxs = np.squeeze(np.max(landmarks, axis=0))
    mins = np.squeeze(np.min(landmarks, axis=0))
    corner_ul = np.array([mins[0], 1 - maxs[1]])
    corner_br = np.array([maxs[0], 1 - mins[1]])

    for i in range(2):
        corner_ul[i] = int(corner_ul[i] * image_size[i])
        corner_br[i] = int(corner_br[i] * image_size[i])

    corner_ul = corner_ul.astype(np.int32)
    corner_br = corner_br.astype(np.int32)
    return corner_ul, corner_br


def main():
    model = keras.models.load_model("models/FER_finetuned.keras")
    print("Model loaded!")
    mp_detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    cam = CVCamera(recording_res=HIGHRES_SIZE, index_cam=1)
    cam.start()

    now = 0
    last = 0
    prediction = 0

    while True:
        image_rgb = cam.read_frame()
        now = time.time()
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_detector.process(image_rgb)
        _, landmarks = face_get_XYZ(results=results, image_rgb=None)
        corner_ul, corner_br = get_face_corners(landmarks, HIGHRES_SIZE)

        # Process the image to the model
        if now - last > 0.5:

            if landmarks.sum() == 0:
                continue

            landmarks = landmarks.astype(np.float32)

            # Apply normalizations
            landmarks = normalize_L0(landmarks)
            landmarks = normalize_size(landmarks)

            # Add axis for batch (axis 0) and channels (axis 3)
            landmarks = np.expand_dims(landmarks, (0, 3))

            # Obtain model's prediction
            prediction = np.argmax(model(landmarks))

            last = time.time()

        # Plot emotion label
        cv2.putText(
            image_rgb,
            f"Emotion: {my_recorded_classes[prediction]}",
            org=(corner_ul[0], corner_ul[1] - 5),
            fontFace=2,
            fontScale=0.65,
            color=RED,
        )

        # Plot rectangle around face
        cv2.rectangle(image_rgb, corner_ul, corner_br, color=RED, thickness=2)

        cv2.imshow("Emotions classifier", image_rgb)
        key = cv2.waitKey(int(1 / FPS * 1000)) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
