import cv2
import mediapipe as mp
import joblib

model=joblib.load("model.pkl")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)


LIP_LANDMARKS = list(range(78, 89)) + list(range(308, 319))
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
           print("Ignoring empty camera frame.")
           continue

  
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        #face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                lip_coords = [
                    (face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y)
                    for idx in LIP_LANDMARKS
                ]
                flat_coords = [coord for point in lip_coords for coord in point]
                prediction = model.predict([flat_coords])  # predict the smile status
                print("Smile" if prediction == 1 else "No Smile")
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                
            #   print(list(face_landmarks.landmark)) for testing
            
                # flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == ord("q"):
           break
cap.release()