import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import functional as F
from torchvision.transforms import Resize
import numpy as np
from time import time
import os
import pathlib
from pathlib import Path
import csv
from datetime import datetime
import base64
class Detection:
    """
    Class implements Yolo5 model to make inferences on a video stream using OpenCV2.
    """

    def __init__(self, capture_index, model_name):
        """
        Initializes the class with video capture index and YOLOv5 model file.
        :param capture_index: Index of the video capture device.
        :param model_name: Name of the YOLOv5 model file.
        """
        self.capture_index = capture_index
        self.object_detector = self.load_object_detector(model_name)
        self.face_detector = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40)
        self.face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()
        self.classes = self.object_detector.names
        self.device = "cpu"
        print("Using Device:", self.device)
        # CSV file to store recognition data
        self.recognition_csv_file = "recognition_data.csv"
        self.fine_csv_file="fine_details.csv"
        self.create_csv_files()
        
    def create_csv_files(self):
        """
        Create CSV files if they don't exist and write headers.
        """
        if not os.path.exists(self.recognition_csv_file):
            with open(self.recognition_csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Time", "Location"])

        if not os.path.exists(self.fine_csv_file):
            with open(self.fine_csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Name", "Time", "Location", "Fine Amount"])

    def append_to_csv(self, name, camera_index):
        """
        Append recognition data to the CSV file.
        :param name: Name of the recognized person.
        :param camera_index: Index of the camera where the recognition occurred.
        """
        with open(self.recognition_csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, datetime.now(), camera_index])
            
    def append_fine_details(self, name, camera_index, fine_amount):
        """
        Append fine details to the fine CSV file.
        :param name: Name of the person without an ID card.
        :param camera_index: Index of the camera where the person was detected.
        :param fine_amount: Amount of fine to be charged.
        """
        with open(self.fine_csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, datetime.now(), camera_index, fine_amount])

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame for prediction.
        :return: OpenCV2 video capture object.
        """
        return cv2.VideoCapture(self.capture_index)

    def load_object_detector(self, model_name):
        """
        Loads YOLOv5 model from a specified path.
        :param model_name: Name of the YOLOv5 model file.
        :return: Trained PyTorch model.
        """
        model_path = os.path.join("object_detect", model_name)
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def detect_objects(self, frame):
        """
        Detects objects in a frame using the YOLOv5 model.
        :param frame: Input frame in numpy array format.
        :return: Labels and coordinates of objects detected by the model in the frame.
        """
        self.object_detector.to(self.device)
        results = self.object_detector(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def recognize_faces(self, frame, camera_index):
        """
        Recognizes faces in a frame using MTCNN and InceptionResnetV1 models.
        :param frame: Input frame in numpy array format.
        :param camera_index: Index of the camera where the frame is captured.
        :return: Detected faces and their embeddings.
        """
        frame_pil = F.to_pil_image(frame)
        boxes, _ = self.face_detector.detect(frame_pil)
        faces = []
        embeddings = []
        identity = "Not detected"  # Initialize identity variable
        if boxes is not None:
            for i, box in enumerate(boxes):
                # Crop face from the frame
                face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                
                # Convert face to PIL image
                face_pil = F.to_pil_image(face)
                resize_transform = Resize((160, 160))
                # Resize face image
                face_pil_resized = resize_transform(face_pil)
                data = torch.load('integrated\data.pt')
                embedding_list, name_list = data
                # Extract embeddings
                emb = self.face_recognizer(F.to_tensor(face_pil_resized).unsqueeze(0))
                embedding_list, name_list = data
                # Compare embeddings with known embeddings
                min_dist = float('inf')
                identity = None
                for idx, known_emb in enumerate(embedding_list):
                    dist = torch.dist(emb, known_emb)
                    if dist < min_dist:
                        min_dist = dist
                        identity = name_list[idx]
                if identity:
                    self.append_to_csv(identity, camera_index)  # Store the recognition data
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, identity, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return identity
    
    def class_to_label(self, x):
        """
        Converts a numeric label to a corresponding string label.
        :param x: Numeric label.
        :return: Corresponding string label.
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Plots bounding boxes and labels on a frame.
        :param results: Contains labels and coordinates predicted by the model on the frame.
        :param frame: Frame with bounding boxes and labels plotted on it.
        :return: Frame with bounding boxes and labels plotted on it.
        """
        labels, cord = results
        if len(labels) == 0:
            return frame
        
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5:
                object_name = self.class_to_label(labels[i])
                print("Detected object:", object_name)
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, object_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    

    def vidcap(self):
        """
        Executes the object detection and face recognition on the video stream.
        """
        cap = self.get_video_capture()
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            assert ret

            frame = cv2.resize(frame, (416, 416))

            start_time = time()
            # Detect faces and their embeddings
            camera_index = 1
            face_results = self.recognize_faces(frame, camera_index)
            print(face_results)
            confidence_threshold = 0.5
            if face_results:
                # If faces are detected, pass the frame through object detection
                object_results = self.detect_objects(frame)
                labels, cords = object_results  # Extracting labels from object_results
                any_valid_detection = False
                for i in range(len(labels)):
                    if self.class_to_label(labels[i]) in ['Lanyard', 'Cards'] and cords[i][4] >= confidence_threshold:
                        any_valid_detection = True
                        break  # Exit loop if a valid detection is found

                if not any_valid_detection:
                    # No valid detections (lanyard or card) above confidence threshold
                    missing_item = "Both lanyard and card"
                    fine_amount = 100
                    reason = "Missing required items (lanyard and card)"
                    self.append_fine_details(face_results, 1, 50)
                    print(f"No lanyard or card detected with high confidence")
                else:
                    # If objects are detected, plot them on the frame
                    frame = self.plot_boxes(object_results, frame)

            end_time = time()

            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            return frame



# Create a new object but don't invoke it here.
#detector = Detection(capture_index=0, model_name='best.pt')
