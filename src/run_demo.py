import keras
import cv2
from cameras import CVCamera

HIGHRES_SIZE = (1280, 720)
LARGE_SIZE = (640, 480)
SMALL_SIZE = (320, 200)


def main():
    model = keras.models.load_model("models/cnn.keras")
    cam = CVCamera(recording_res=LARGE_SIZE, index_cam=0)
    cam.start()

    while True:
        image = cam.read_frame()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imshow(image_rgb)


if __name__ == "__main__":
    main()
