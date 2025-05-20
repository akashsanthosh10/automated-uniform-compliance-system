from flask import Flask, render_template, Response, request
import cv2
from detection import Detection
import pathlib
import sqlite3
app = Flask(__name__)
pathlib.PosixPath = pathlib.WindowsPath
detector = Detection(capture_index=0,model_name='best.pt')



@app.route('/fine_details')
def fine_details():
    # Create a new connection within the current thread
    conn = sqlite3.connect('recognition_data.db')
    
    # Fetch data from the fine_details table
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM fine_details")
    fine_details_data = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return render_template('fine_details.html', fine_details_data=fine_details_data)


@app.route('/tracking_info')
def tracking_info():
    # Create a new connection within the current thread
    conn = sqlite3.connect('recognition_data.db')
    
    # Fetch data from the tracking table
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tracking")
    tracking_data = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return render_template('tracking_info.html', tracking_data=tracking_data)



def generate_frames():
    for frame in detector.vidcap():
        yield frame


@app.route('/')
def index():
    message = None  # Initialize message as None
    return render_template('index.html', message=message)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":
    app.run(debug=True)
