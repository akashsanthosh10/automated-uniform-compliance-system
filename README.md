#  Automated Uniform Compliance System

A automated uniform compliance system built with Flask, YOLOv5, and FaceNet to detect and recognize students from video feeds, log tracking data, issue fines for uniform/code violations, and notify concerned personnel via email.

## 📸 Features

- Real-time video streaming with YOLOv5-based object detection
- Face detection and recognition using MTCNN + FaceNet (InceptionResnetV1)
- Logging of student tracking data with timestamp and camera index
- Fine issuance with email alerts if a student violates predefined rules
- Cooldown mechanism to avoid duplicate fines
- SQLite database integration for persistent storage
- Web interface for live stream, fine details, and tracking info

## 🧰 Technologies Used

- **Flask** – Web application framework
- **YOLOv5** – Object detection model
- **MTCNN + FaceNet** – Face detection and recognition
- **SQLite** – Lightweight relational database
- **OpenCV** – Video capture and frame processing
- **smtplib** – Sending fine notification emails

<!---## 🏗️ Project Structure

```
.
├── app.py                  # Main Flask app
├── detection.py            # Detection and recognition logic
├── templates/
│   ├── index.html
│   ├── fine_details.html
│   └── tracking_info.html
├── static/                 # Optional for CSS/JS if needed
├── object_detect/
│   └── best.pt             # Custom YOLOv5 trained model
├── integrated/
│   └── data.pt             # Face embeddings and names
└── recognition_data.db     # SQLite database file
```
-->
## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/akashsanthosh10/automated-uniform-compliance-system
cd automated-uniform-compliance-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Include these in `requirements.txt`:
```txt
flask
opencv-python
torch
facenet-pytorch
torchvision
numpy
```

### 3. Add Required Files

- `object_detect/best.pt`: Your YOLOv5 trained weights.
- `integrated/data.pt`: Precomputed embeddings with corresponding names.
- Make sure `recognition_data.db` is present, or it will be created automatically.

### 4. Configure Email Settings

Update `Detection.sentemail()` in `detection.py` with your actual email credentials:

```python
sender_email = "your_email@gmail.com"
receiver_email = "teacher_email@example.com"
password = "your_email_password"
```

You may need to enable **App Passwords** or **Less Secure App Access** in your Gmail account.

### 5. Run the Application

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000/` in your browser.

## 📊 Web Interface

- `/` – Live video feed
- `/fine_details` – View fine history
- `/tracking_info` – View student tracking logs

## 🧪 Sample Use Cases

- Monitor students during class time
- Fine students not wearing ID cards or improper uniform
- Track students roaming in unauthorized areas

## 📬 Future Improvements

- Admin panel for fine management
- Camera location mapping
- Multi-camera integration
- Student database and authentication

## Screenshots

![Image](https://github.com/user-attachments/assets/2240049f-c6a1-4f04-bbc8-b57c07324eb4)
![Image](https://github.com/user-attachments/assets/519370cb-5a0e-41d4-8b62-f0597b4c5ab2)
![Image](https://github.com/user-attachments/assets/bfe37e7c-4092-4c2b-90ec-562df57fd22c)
![Image](https://github.com/user-attachments/assets/d196235f-a22b-4bff-9f59-abb63b525f77)
![Image](https://github.com/user-attachments/assets/e4bf3292-5e6f-4041-b9b1-7a37b999faa0)
![Image](https://github.com/user-attachments/assets/58061451-fc68-416a-b680-88ccbdc2f7a3)

## 📝 License

This project is licensed under the MIT License.

---

**Note**: This project is intended for academic/research purposes. Ensure it complies with privacy and legal policies before deployment.
