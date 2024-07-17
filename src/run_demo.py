import keras
import cv2
import time
import numpy as np
import mediapipe as mp
from cameras import CVCamera
from landmarks_utils import face_get_XYZ

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

classes = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]


def main():
    model = keras.models.load_model("models/model-best.h5")
    print("Model loaded!")
    mp_detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    cam = CVCamera(recording_res=LARGE_SIZE, index_cam=1)
    cam.start()

    now = 0
    last = 0

    while True:
        image_rgb = cam.read_frame()
        now = time.time()
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imshow("camera", image_rgb)

        # Process the image to the model
        if now - last > 0.2:
            results = mp_detector.process(image_rgb)
            _, landmarks = face_get_XYZ(results)
            prediction = model.predict(np.array(landmarks))

            cv2.putText(
                image_rgb,
                f"Emotion: {classes[prediction]}",
                org=(20, 40),
                fontFace=2,
                fontScale=0.65,
                color=RED,
            )

            last = time.time()

        key = cv2.waitKey(int(1 / FPS * 1000)) & 0xFF

        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
