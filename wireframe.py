import cv2
import mediapipe as mp
import numpy as np

vision = mp.tasks.vision
RunningMode = vision.RunningMode
draw = vision.drawing_utils
styles = vision.drawing_styles

HandConn = vision.HandLandmarksConnections
FaceConn = vision.FaceLandmarksConnections
PoseConn = vision.PoseLandmarksConnections

hand_opts = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
)
face_opts = vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=RunningMode.VIDEO,
)
pose_opts = vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=RunningMode.VIDEO,
)

hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)
face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

cap = cv2.VideoCapture(0)
timestamp_ms = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

    annotated = np.copy(rgb)

    for hand_lms in hand_result.hand_landmarks:
        draw.draw_landmarks(
            annotated, hand_lms,
            HandConn.HAND_CONNECTIONS,
            styles.get_default_hand_landmarks_style(),
            styles.get_default_hand_connections_style(),
        )

    for face_lms in face_result.face_landmarks:
        draw.draw_landmarks(
            annotated, face_lms,
            FaceConn.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=styles.get_default_face_mesh_tesselation_style(),
        )
        draw.draw_landmarks(
            annotated, face_lms,
            FaceConn.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=styles.get_default_face_mesh_contours_style(),
        )

    for pose_lms in pose_result.pose_landmarks:
        draw.draw_landmarks(
            annotated, pose_lms,
            PoseConn.POSE_LANDMARKS,
            styles.get_default_pose_landmarks_style(),
        )

    out = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imshow("Wireframe", out)

    timestamp_ms += 33
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
