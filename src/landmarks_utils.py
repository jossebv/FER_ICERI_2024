import mediapipe as mp
import numpy as np


def hand_get_XYZ(results, image_rgb):

    landmark_values = [[0, 0] for _ in range(21)]

    # esto para dibujar
    mp_drawing = mp.solutions.drawing_utils

    image_height, image_width, _ = image_rgb.shape

    for hand_landmarks in results.multi_hand_landmarks:
        # Draw hand landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(
            image_rgb,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(
                color=(255, 255, 0), thickness=4, circle_radius=5
            ),
            mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=4),
        )

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x  # * image_width
            y = (
                1 - hand_landmarks.landmark[i].y
            )  # * image_height (1-y for providing same shape as in image)
            landmark_values[i] = [x, y]

    return image_rgb, landmark_values


def face_get_XYZ(results, image_rgb=None):

    landmark_values = [[0, 0] for _ in range(478)]

    if results.multi_face_landmarks is None:
        return image_rgb, None

    for face_landmarks in results.multi_face_landmarks:
        for i in range(len(face_landmarks.landmark)):
            x = face_landmarks.landmark[i].x
            y = 1 - face_landmarks.landmark[i].y
            landmark_values[i] = [x, y]

        if image_rgb is None:
            landmarks = np.array(landmark_values)
            return None, landmarks

        mp.solutions.drawing_utils.draw_landmarks(
            image=image_rgb,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=image_rgb,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=image_rgb,
            landmark_list=face_landmarks,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return image_rgb, landmark_values
