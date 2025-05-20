import timeit
#import threading
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import functional as F
from torchvision.transforms import Resize
import numpy as np
#import pathlib
import sqlite3
from datetime import datetime, timedelta
#import csv
from datetime import datetime
import os
#import queue
import smtplib, ssl
from email.message import EmailMessage
class Detection:
    """
    Class implements Yolo5 model to make inferences on a video stream using OpenCV2.
    """

    def __init__(self, capture_index: int, model_name: str):
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
        self.fine_cooldown = timedelta(minutes=1)  # Cooldown period for fines (5 minutes in this example)
        self.recent_fine_window = timedelta(hours=1)  # Time window to consider for recent fines (1 hour in this example)
        self.last_fine_time = {}
        self.conn = sqlite3.connect('recognition_data.db')
        self.create_tables()
    def sentemail(self,message):
        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"
        sender_email = "admmin@gmail.com"  
        receiver_email = "teacher@gmail.com"  
        password = "---"
        Subject="Fine message"
        body = message
        em=EmailMessage()
        em['From']=sender_email
        em['To']=receiver_email
        em['Subject']=Subject
        em.set_content(body)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port,context=context) as smtp:
            smtp.login(sender_email, password)
            smtp.sendmail(sender_email, receiver_email, em.as_string())
            print("Successfully Sent email")

    def create_tables(self):
        # Create tables if they don't exist


            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS tracking (
                    trackid INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    time TEXT NOT NULL,
                    camera_no INTEGER NOT NULL,
                    FOREIGN KEY (name) REFERENCES person(name)
                )
            ''')

            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS fine_details (
                    fine_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    time TEXT NOT NULL,
                    camera_no INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    FOREIGN KEY (name) REFERENCES person(name)
                )
            ''')

    def log_tracking(self, name, camera_index):
        conn = sqlite3.connect('recognition_data.db')
        timestamp = datetime.now()
        with conn:
            conn.execute('''
                INSERT INTO tracking (name, time, camera_no)
                VALUES (?, ?, ?)
            ''', (name,timestamp.strftime('%Y-%m-%d %H:%M:%S'), camera_index))
    def append_fine_details(self, name, description, camera_no, amount):
        """
        Logs fine details into the fine_details table if a certain cooldown period has elapsed since the last fine for the person.

        :param name: Name of the person fined.
        :param description: Description of the fine reason.
        :param camera_no: Index of the camera where the fine occurred.
        :param amount: Fine amount.
        """
        message=""
        conn = sqlite3.connect('recognition_data.db')
        timestamp = datetime.now()
        last_fine_time = self.last_fine_time.get(name, datetime.min)

        if timestamp - last_fine_time > self.fine_cooldown and not self.has_recent_fine(name):
            # If cooldown period has elapsed since last fine and person hasn't been fined recently
            with conn:
                conn.execute('''
                    INSERT INTO fine_details (name, description, time, camera_no, amount)
                    VALUES (?, ?, ?, ?, ?)
                ''', (name, description, timestamp.strftime('%Y-%m-%d %H:%M:%S'), camera_no, amount))
            message = f"A fine of {amount} was issued to {name} for {description} on camera number {camera_no} at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}."
            self.sentemail(message)
            # Update the last fine time for the person
            self.last_fine_time[name] = timestamp

        return message
    def has_recent_fine(self, name):
        """
        Checks if the person has been fined within the recent fine window.

        :param name: Name of the person.
        :return: True if the person has been fined recently, False otherwise.
        """
        conn = sqlite3.connect('recognition_data.db')
        recent_window = datetime.now() - self.recent_fine_window
        with conn:
            result = conn.execute('''
                SELECT COUNT(*) FROM fine_details
                WHERE name = ? AND time >= ?
            ''', (name, recent_window.strftime('%Y-%m-%d %H:%M:%S'))).fetchone()
        return result[0] > 0

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
        frame = cv2.resize(frame, (320, 240))
        self.object_detector.to(self.device)
        results = self.object_detector(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def recognize_faces(self, frame, camera_index):
        threshold=0.3
        frame_pil = F.to_pil_image(frame)
        boxes, _ = self.face_detector.detect(frame_pil)
        faces = []
        embeddings = []
        data = torch.load('integrated\data.pt')
        embedding_list, name_list = data
        identities = []  # List to store identities for each detected face
        
        if boxes is not None:
            for i, box in enumerate(boxes):
                # Crop face from the frame
                face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                
                # Convert face to PIL image
                face_pil = F.to_pil_image(face)
                resize_transform = Resize((160, 160))
                # Resize face image
                face_pil_resized = resize_transform(face_pil)
                
                # Extract embeddings
                emb = self.face_recognizer(F.to_tensor(face_pil_resized).unsqueeze(0))
                
                # Compare embeddings with known embeddings
                min_dist = float('inf')
                identity = None
                for idx, known_emb in enumerate(embedding_list):
                    dist = torch.dist(emb, known_emb)
                    if dist < min_dist:
                        min_dist = dist
                        identity = name_list[idx]
                
                if min_dist > threshold:
                    self.log_tracking(identity, camera_index)  # Store the recognition data
                else:
                    identity = "Unknown"  # Assign as unknown if no suitable match found
                identities.append(identity)

                # Visualize the detection and identity
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
                cv2.putText(frame, identity, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return identities

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
            #print("Not detected")
            return frame
        
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.8:
                object_name = self.class_to_label(labels[i])
                #print("Detected object:", object_name)
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
                cv2.putText(frame, object_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 1)
        return frame


    def vidcap(self, message=None):
        """
        Executes the object detection and face recognition on the video stream.
        """
        cap = self.get_video_capture()
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            assert ret

            frame = cv2.resize(frame, (320, 240))

            # Perform face recognition on the frame
            camera_index = 1
            face_results = self.recognize_faces(frame, camera_index)
            print(face_results)
            confidence_threshold = 0.8

            if face_results:
                # If faces are detected, pass the frame through object detection
                object_results = self.detect_objects(frame)
                labels, cords = object_results  # Extracting labels from object_results
                any_valid_detection = False
                for i in range(len(labels)):
                    row=cords[i]
                    print(row,self.class_to_label(labels[i]))
                    if row[4]>=confidence_threshold:
                        any_valid_detection = True
                        break  # Exit loop if a valid detection is found

                if not any_valid_detection:
                    # No valid detections (lanyard or card) above confidence threshold
                    missing_item = "Both lanyard and card"
                    fine_amount = 100
                    reason = "Missing ID card"
                    # self.append_fine_details(missing_item, fine_amount, reason)
                    message=self.append_fine_details(face_results[0],reason, self.capture_index, fine_amount)
                    print(f"No lanyard or card detected with high confidence")
                else:
                    # If objects are detected, plot them on the frame
                    frame = self.plot_boxes(object_results, frame)
                    #pass  

            start_time = timeit.default_timer()

            """# Display the processed frame (optional)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break"""

            end_time = timeit.default_timer()

            # fps = 1 / np.round(end_time - start_time, 2)
            # cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()
        cv2.destroyAllWindows()


# Create a new object but don't invoke it here.
#detector = Detection(capture_index=0, model_name='best.pt')