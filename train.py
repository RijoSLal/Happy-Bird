import cv2
import mediapipe as mp
import csv
import os

# initialize MediaPipe Face Mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles

# important lip landmarks (this is may not be accurate since no documentation is available)
LIP_LANDMARKS = list(range(78, 89)) + list(range(308, 319))

# dir for images (smile and non-smile)
image_smile_dir = "smile/"
image_no_smile_dir = "non_smile/"

# video files for smile and non-smile videos
video_smile = "smile.mp4"
video_no_smile = "nosmile.mp4"

cap_smile = cv2.VideoCapture(video_smile)
cap_no_smile = cv2.VideoCapture(video_no_smile)

# set video frame size  (optional)
cap_smile.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap_smile.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
cap_no_smile.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap_no_smile.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

dataset = []

# to process the video and collect the data
def process_video(cap, label):
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # extract lip coordinates (for mouth area)
                    lip_coords = [
                        (face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y)
                        for idx in LIP_LANDMARKS
                    ]

                    flat_coords = [coord for point in lip_coords for coord in point]
                    dataset.append(flat_coords + [label])

                    # draw the landmarks on the image for visualization (optional just to know its working)
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("Processing Video", cv2.flip(image, 1))

            if cv2.waitKey(24) & 0xFF == ord("q"):
                break

# to process images in a folder
def process_images(image_dir, label):
    if not os.listdir(image_dir):  # check for empty folder
        print(f"Warning: The folder '{image_dir}' is empty. Skipping image processing.")
        return

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        for filename in os.listdir(image_dir):
            img_path = os.path.join(image_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # extract lip coordinates (for mouth area)
                    lip_coords = [
                        (face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y)
                        for idx in LIP_LANDMARKS
                    ]

                    flat_coords = [coord for point in lip_coords for coord in point]
                    dataset.append(flat_coords + [label])

# process smile and non-smile videos
print("Processing smile video...")
process_video(cap_smile, label=1)  # 1 for smile
print("Processing non-smile video...")
process_video(cap_no_smile, label=0)  # 0 for no smile

# process smile and non-smile images
print("Processing smile images...")
process_images(image_smile_dir, label=1)
print("Processing non-smile images...")
process_images(image_no_smile_dir, label=0)

cap_smile.release()
cap_no_smile.release()
cv2.destroyAllWindows()

# save dataset to CSV
csv_file = "lip_landmarks_dataset.csv"
header = [f"x{i}" for i in range(len(LIP_LANDMARKS) * 2)] + ["label"]

with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(dataset)

print(f"Dataset saved to {csv_file}")











































# import cv2
# import mediapipe as mp
# import csv

# # Initialize MediaPipe Face Mesh
# mp_drawing = mp.solutions.drawing_utils
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing_styles = mp.solutions.drawing_styles

# # Important lip landmarks (polygonal mouth area)
# LIP_LANDMARKS = list(range(78, 89)) + list(range(308, 319))

# # Initialize video capture (for smile and non-smile videos)
# video_smile = "smile.mp4"
# video_no_smile = "nosmile.mp4"

# cap_smile = cv2.VideoCapture(video_smile)
# cap_no_smile = cv2.VideoCapture(video_no_smile)

# # Set video frame size
# cap_smile.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
# cap_smile.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
# cap_no_smile.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
# cap_no_smile.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

# dataset = []

# # Function to process the video and collect the data
# def process_video(cap, label):
#     with mp_face_mesh.FaceMesh(
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     ) as face_mesh:
#         while cap.isOpened():
#             success, image = cap.read()
#             if not success:
#                 break

#             image.flags.writeable = False
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(image)

#             if results.multi_face_landmarks:
#                 for face_landmarks in results.multi_face_landmarks:
#                     # Extract lip coordinates (for mouth area)
#                     lip_coords = [
#                         (face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y)
#                         for idx in LIP_LANDMARKS
#                     ]

#                     # Flatten the coordinates (flattening for the model input)
#                     flat_coords = [coord for point in lip_coords for coord in point]
#                     dataset.append(flat_coords + [label])

#                     # Draw the landmarks on the image for visualization
#                     mp_drawing.draw_landmarks(
#                         image=image,
#                         landmark_list=face_landmarks,
#                         connections=mp_face_mesh.FACEMESH_TESSELATION,
#                         landmark_drawing_spec=None,
#                         connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
#                     )

#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#             cv2.imshow("Processing Video", cv2.flip(image, 1))

#             if cv2.waitKey(24) & 0xFF == ord("q"):
#                 break

# print("Processing smile video...")
# process_video(cap_smile, label=1)  # 1 for smile
# print("Processing non-smile video...")
# process_video(cap_no_smile, label=0)  # 0 for no smile

# cap_smile.release()
# cap_no_smile.release()
# cv2.destroyAllWindows()

# # Save dataset to CSV
# csv_file = "lip_landmarks_dataset.csv"
# header = [f"x{i}" for i in range(len(LIP_LANDMARKS) * 2)] + ["label"]

# with open(csv_file, mode="w", newline="") as file:
#     writer = csv.writer(file)
#     writer.writerow(header)
#     writer.writerows(dataset)

# print(f"Dataset saved to {csv_file}")
