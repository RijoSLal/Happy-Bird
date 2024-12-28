from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
import mediapipe as mp
import joblib



app = FastAPI()


templates = Jinja2Templates(directory="templates")

model=joblib.load("model.pkl")



cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

smile_status = {"status": None}

def generate_frames():
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
                    smile_status["status"] = prediction 
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    


            _, buffer = cv2.imencode(".jpg",image)  
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """main HTML page"""
    return templates.TemplateResponse("flappy.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    """return the video stream as a response."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/smile_status")
def get_smile_status():
    """Serve real-time smile status updates."""
    def smile_event_stream():
        while True:
            yield f"data: {smile_status['status']}\n\n"

    return StreamingResponse(smile_event_stream(), media_type="text/event-stream")



"run  uvicorn fast:app --reload to run this main file"