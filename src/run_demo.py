print("Loading dependencies... (It might take a while the first time)")
import keras
import cv2
import time
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from cameras import CVCamera, PICamera
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

my_recorded_classes = [
    "Angry",
    "Happy",
    "Sad",
    "Surprise",
]  # Remember they must keep the same order than the used in training


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


def draw_results(image, results, face_corners=None):
    """
    Draws the results and graphics on top of the camera's recording.

    Parameters:
        - image: array of the recorded image
        - results: normalized logits obtained from the model
        - face_corners: list with the corners with format [corner_up_left, corner_down_right] == [(x0,y0), (x1,y1)]
    """
    prediction_idx = np.argmax(results)
    probability = results[0, prediction_idx] * 100
    label = my_recorded_classes[prediction_idx]

    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)

    # emoji_font = ImageFont.truetype("NotoColorEmoji.ttf", 48)
    text_font = ImageFont.truetype("Arial.ttf", 16)
    draw.text(
        (face_corners[0][0], face_corners[0][1] - 20),
        f"Emotion: {label} ({probability:.2f}%)",
        font=text_font,
        fill=(0, 0, 255, 255),
    )

    if face_corners is not None:
        draw.rectangle(face_corners, outline=(0, 0, 255, 255))

    image_rgb = np.array(image_pil)
    return image_rgb


def main():
    # Load model to use
    print("Loading model...")
    model = keras.models.load_model("models/FER_finetuned.keras")
    print("Model loaded!")

    # Load mediapipe Face Mesh utility for landmark extraction
    mp_detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )

    # Start camera, use CVCamera if working on a laptop and PICamera in case you are working on a Raspberry PI
    cam = CVCamera(recording_res=HIGHRES_SIZE, index_cam=1)
    # cam = PICamera(recording_res=HIGHRES_SIZE)

    cam.start()

    now = 0
    last = 0
    prediction = 0

    while True:
        image_rgb = cam.read_frame()
        if image_rgb is None:
            # Depending the setup, the camera might need approval to activate, so wait until we start receiving images.
            continue

        now = time.time()
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = mp_detector.process(image_rgb)
        _, landmarks = face_get_XYZ(results=results, image_rgb=None)
        corner_ul, corner_br = get_face_corners(landmarks, HIGHRES_SIZE)

        # Process the image to the model every 0.5s
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
            predictions = model(landmarks)

            last = time.time()

        # Plot emotion label
        # cv2.putText(
        #     image_rgb,
        #     f"Emotion: {my_recorded_classes[prediction]}",
        #     org=(corner_ul[0], corner_ul[1] - 5),
        #     fontFace=2,
        #     fontScale=0.65,
        #     color=RED,
        # )

        # # Plot rectangle around face
        # cv2.rectangle(image_rgb, corner_ul, corner_br, color=RED, thickness=2)

        image_rgb = draw_results(
            image_rgb, predictions, face_corners=[tuple(corner_ul), tuple(corner_br)]
        )

        cv2.imshow("Emotions classifier", image_rgb)
        key = cv2.waitKey(int(1 / FPS * 1000)) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
