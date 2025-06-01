from flask import Flask, render_template, Response, request, send_file, redirect, url_for, flash, session,jsonify
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import keras
from keras.models import load_model
import cv2
import numpy as np
import os
import zipfile
import time
from threading import Lock, Thread
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, EmailField
from wtforms.validators import DataRequired, Email, EqualTo, Length
import dropbox
from datetime import datetime, timedelta  # Add timedelta to the imports
import math
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
import pytz  # Import for handling time zones
import logging
from math import ceil, sqrt
import sqlite3
import webbrowser
import sys



# Configure logging


# MongoDB Configuration
client = MongoClient('mongodb://localhost:27017/')  # Connect to MongoDB
db_mongo = client['framesdb']  # Database name
frames_collection = db_mongo['frames']  # Collection name for metadata





# Load Keras model
model_path = r'C:\Users\User\Desktop\NEW MONGO - Copy - Copy\vio_vedio.h5'
model = keras.models.load_model(model_path)



# Flask app configuration
if getattr(sys, 'frozen', False):
    # Running as an executable
    STATIC_FOLDER = r"D:\pyinstaller_dist\static"
    app = Flask(__name__, static_folder=STATIC_FOLDER)
else:
    # Running as a Python script
    app = Flask(__name__)
    
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "instance", "database.db")
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DB_PATH}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False



# Database setup
db = SQLAlchemy(app)



# Define User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)



# Initialize Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = XXX
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'vavi3984@gmail.com'
app.config['MAIL_PASSWORD'] = 'XXX'  # Use app password here
app.config['MAIL_DEFAULT_SENDER'] = 'aviverma939@gmail.com'



mail = Mail(app)

# Brevo API Configuration

# Folder paths
FRAME_SAVE_PATH = os.path.join(app.static_folder, 'frames')
os.makedirs(FRAME_SAVE_PATH, exist_ok=True)
FRAMES_PATH = r"C:\\Users\\User\\Desktop\\FRAMES"
os.makedirs(FRAMES_PATH, exist_ok=True)
ARC_PATH = r"C:\\Users\\User\\Desktop\\ARC"
os.makedirs(ARC_PATH, exist_ok=True)

file_path1 = "D:/pyinstaller_dist/camera_links.txt.txt"
if os.path.exists(file_path1):
    last_modified_time = os.path.getmtime(file_path1)
else:
    last_modified_time = 0 
    
RTSP_URLS = None

def load_rtsp_links(fp):
    global RTSP_URLS
    if not os.path.exists(fp):
        return []  # Return empty list if file is missing
    
    with open(fp, "r") as f:
        links = [line.strip() for line in f.readlines() if line.strip()]
    RTSP_URLS=links


# RTSP streams configuration
  # Restart streams


# Shared variables for the live streams
streams = {}  # Dictionary to store the latest frame and lock for each stream
capture_threads = {}  # Dictionary to manage capture threads



# Dropbox setup
DROPBOX_ACCESS_TOKEN = 'sl.u.AFnkt11JVO2lXVpQjis-8232XLaNEReuhCcjInFwHCUKXbLxQmrtEZXX4nRaNUvI-20QKZv0wXm6EP1QDARhZxZ_99iX1ZQs2nX9afVfive_QvzeB0p07KEfi09tGjTgLDj9EAWw-exuZDsKUX6sHjPsHONElSWZuHiHt7MeLP6m8Ox3MSVzF3kzUY5xfBeJU7htH6e3LP8UOEjwPej5ZS6ySc8lIlGyMhMRj7ahaFulmlO7DIhuzSrWDAReLj6ieI0ag6rqKst7eOsnCXYhnP63B53W5VP8pyAtnqJrqRXCzyNjbDjbV_nXl2-E13RKwXqEOr4kFgQBS3b07_yexIyS-zgi8rG28vdqyRb7xzFknD_d93mjE4Xgq6mI2KJRczwxH-3_hg7Sxut2-m86xihGPKbRZnxM17lQqCeC0qjS7rNWAIMJDyIOUucSazPqCdfdWIyclblGjuCcpJGy-MPMFtGG_qj6-EXB34-XFBfb7wPBiw-eRYy4DlTBJYIsNCUo3I-kNx5zlNoJXqPcD_OzMcSrdX90McftwBaDf_uZCST55Mh0tg0mca3areKk8uv6ImV177ozm-qxIxs-zX8_vc2-IBe8TvQT2CwGVjHIM26xBbDgdjPF_IsJXKrXt19OM3-T4yGQgbe9vYylt6OzZDcTXJrJYn7yfjuQn5-CexLilCnCJZF5o0baFreubzDLp80h0Cbp3xxgG3oc0grymePB5unS-_6RAmjsHF7_koXk5XLjNnPcfWxK2m-d-LzNzEkvpbSh9ojP_MbBvR1q6dTGeK1OqzbxmM-a2ILQAJqGkmmA_yTZAWZ1Iron0hbWHpNw0xg-BkswqUjtsjPf9ql017uFn7AWhDbo4Gu8XO_697C9zTOnSQYiumrA8V7QzewMbtMS3NLfQ7Qf0ph3GjgZiLpFdWie1IfmwcZ5dqS45XN6GalE2FqUQ1IqknxcNfaX13HtnwDTTJo1MobYy-VOb0aCXs4T89F_YeuKCVSILIEcWJLnfJfhf-eozv1eRbs5Ux0TpGWggxh1KlcZ6sFZIFLve4NVsEsxHkwP_CPDgqps0fDT-Zdeq7bLOHRpcn-H_3Y8nfRFvxCMNT8AVfAiXfVSWgNWdSBm6juT7IQG0M5xm5JACbgtiVC0jPUiCgdl1RvcJ2pwH9W1vk0pJSmT4Ig6JmOPZxiUATWYKjQjdx1Ap6-MhPItmLnVDXzVE_6i8zybrovl7Jt6g05KC_rYUClnu2QrUk-249tSxcL5OHuu7Qp9_RH88F3hlc-59KEpbo4CbbjEr21_eD6YDRp2BVPhVndyve6t6wvcRABF9FgEO-8ij8UGkBsa7Tz53DAXErBHmZSNrcm7Th6eH3L4d4n-1xuXQ9xL_wxV8dbfYDr-RjWJte19RgA0Lhr46CgMpInwUOWBC4aSTt6h'
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)



# Function to detect violent content in frames
def detect_violence(frame, threshold=0.2):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    preds = model.predict(img)
    violent = preds[0] > threshold
    print(f"Violence detection for frame: {violent}")  # Debug line
    return violent

def upload_to_dropbox(frame_path, frame_name):
    dropbox_path = f'/frames/{frame_name}'
    with open(frame_path, 'rb') as file:
        dbx.files_upload(file.read(), dropbox_path, mode=dropbox.files.WriteMode('overwrite'))
    return f"File uploaded to Dropbox: {dropbox_path}"


# Function to save frames in both static/frames and the external FRAMES folder
def save_frame_in_multiple_locations(frame, stream_id):
    # Get current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Format: 2025-01-20_12-30-00
    # Construct the frame name with stream ID and current time
    frame_name = f"stream{stream_id}_{current_time}.jpg"
    # Define paths
    frame_path_static = os.path.join(FRAME_SAVE_PATH, frame_name)
    frame_path_frames = os.path.join(FRAMES_PATH, frame_name)
    # Save frame and log results
    cv2.imwrite(frame_path_static, frame)
    cv2.imwrite(frame_path_frames, frame)
    print(f"Saved frame {frame_name} to {FRAME_SAVE_PATH} and {FRAMES_PATH}")  # Debug line


def save_and_upload_frame(frame, stream_id):
    """Save frame locally in a date-based manner with recent frames at the top."""
    # Get current date and time
    current_date = datetime.now().strftime("%Y-%m-%d")  # Folder name
    current_time = datetime.now().strftime("%H-%M-%S")  # File timestamp
    

    # Create a folder for today's date if it doesn't exist
    date_folder_static = os.path.join(FRAME_SAVE_PATH, current_date)
    date_folder_frames = os.path.join(FRAMES_PATH, current_date)

    os.makedirs(date_folder_static, exist_ok=True)
    os.makedirs(date_folder_frames, exist_ok=True)

    # Construct frame name with timestamp
    frame_name = f"stream{stream_id}_{current_time}.jpg"

    # Define local save paths within the date folder
    frame_path_static = os.path.join(date_folder_static, frame_name)
    frame_path_frames = os.path.join(date_folder_frames, frame_name)

    # Save frame locally
    cv2.imwrite(frame_path_static, frame)
    cv2.imwrite(frame_path_frames, frame)
    # Upload to Dropbox and save metadata
    try:
        dropbox_path = f"/frames/{frame_name}"
        with open(frame_path_static, 'rb') as file:
            dbx.files_upload(file.read(), dropbox_path, mode=dropbox.files.WriteMode('overwrite'))

        # Generate Dropbox file URL
        shared_link_metadata = dbx.sharing_create_shared_link_with_settings(dropbox_path)
        file_url = shared_link_metadata.url.replace('?dl=0', '?raw=1')

        # Save metadata to MongoDB
        metadata = {
            "stream_id": stream_id,
            "frame_name": frame_name,
            "file_url": file_url,
            "upload_time": datetime.now()
        }
        frames_collection.insert_one(metadata)
        print(f"Frame {frame_name} saved and uploaded successfully.")
        # Send email with the captured frame as an attachment
        # Add recipients here
        # List of emails to receive the frame
        
    except Exception as e:
        print(f"Error uploading frame {frame_name}: {e}")

import time
import cv2
import numpy as np
from threading import Thread, Lock
from queue import Queue,Empty 
# Function to capture frames with optimized FPS
frame_queue = Queue(maxsize=100)  # Adjust maxsize as needed

def process_frames():
    while True:
        try:
            stream_id, frame, frame_count = frame_queue.get(timeout=1)
            if detect_violence(frame, threshold=0.2):
                save_frame_in_multiple_locations(frame, stream_id)
                save_and_upload_frame(frame, stream_id)
            frame_queue.task_done()
        except Empty:
            continue

def capture_frames(rtsp_url, stream_id):
    global streams
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"Stream {stream_id} ({rtsp_url}) could not be opened.")
        streams[stream_id]["active"] = False
        return

    frame_count = 0
    prev_time = time.time()

    while streams[stream_id]["active"]:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame from Stream {stream_id}.")
            streams[stream_id]["active"] = False
            break

        frame_count += 1
        if not frame_queue.full():
            frame_queue.put((stream_id, frame, frame_count))

        with streams[stream_id]["lock"]:
            streams[stream_id]["latest_frame"] = frame

        elapsed_time = time.time() - prev_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    cap.release()
    print(f"Stream {stream_id} closed.")

def generate_frames(stream_id):
    prev_time = time.time()
    fps_history = []

    while True:
        with streams[stream_id]["lock"]:
            frame = streams[stream_id].get("latest_frame", None)

        if frame is None:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (640, 360))

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        fps_history.append(fps)
        if len(fps_history) > 10:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        cv2.putText(frame, f'FPS: {avg_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def generate_combined_feed():
    global streams     
    while True:
        frames = []
        # Collect the latest frame from each stream
        for stream_id in streams.keys():
            with streams[stream_id]["lock"]:
                frame = streams[stream_id]["latest_frame"]
            if frame is not None:
                frames.append(frame)
            else:
                # Placeholder for streams with no frame
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                frames.append(placeholder)
        if not frames:
            continue
        # Create the grid of frames (adjust this if needed)
        grid_size = int(np.ceil(np.sqrt(len(frames))))  # Square root to determine rows/columns
        blank_frame = np.zeros_like(frames[0])  # Blank frame for padding
        while len(frames) < grid_size**2:
            frames.append(blank_frame)
        rows = []
        for i in range(0, len(frames), grid_size):
            row = cv2.hconcat(frames[i:i+grid_size])
            rows.append(row)
        grid_frame = cv2.vconcat(rows)
        # Encode the grid frame for streaming
        _, buffer = cv2.imencode('.jpg', grid_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def upload_frame_to_dropbox_and_mongo(frame_path, frame_name, user_id):
    # Upload to Dropbox
    dropbox_path = f'/frames/{frame_name}'
    with open(frame_path, 'rb') as file:
        dbx.files_upload(file.read(), dropbox_path, mode=dropbox.files.WriteMode('overwrite'))
   
    # Generate a shared link for the file
    shared_link_metadata = dbx.sharing_create_shared_link_with_settings(dropbox_path)
    file_url = shared_link_metadata.url.replace('?dl=0', '?raw=1')  # Direct download link
    # Save metadata in MongoDB
    metadata = {
        "user_id": user_id,
        "frame_name": frame_name,
        "file_url": file_url,
        "upload_time": datetime.now()
    }
    frames_collection.insert_one(metadata)  # Insert metadata into MongoDB
    return file_url

def delete_frame_from_dropbox_and_mongo(frame_name):
    """Delete frame from Dropbox and its metadata from MongoDB."""
    try:
        # Delete file from Dropbox
        dropbox_path = f"/frames/{frame_name}"  # Path in Dropbox where the frame is stored
        dbx.files_delete_v2(dropbox_path)  # Dropbox API to delete the file
        print(f"File {frame_name} deleted from Dropbox.")

        # Delete metadata from MongoDB
        frames_collection.delete_one({"frame_name": frame_name})  # MongoDB query to delete the metadata
        print(f"Metadata for {frame_name} deleted from MongoDB.")
        
    except Exception as e:
        # If there is an error (e.g., file doesn't exist), print the error
        print(f"Error deleting {frame_name}: {e}")
        
@app.route('/delete_frame/<frame_name>', methods=['POST'])
def delete_frame(frame_name):
    """Route to delete a frame from Dropbox and MongoDB."""
    # Call the function to delete from Dropbox and MongoDB
    delete_frame_from_dropbox_and_mongo(frame_name)

    # Provide feedback to the user
    flash(f"Frame {frame_name} deleted successfully!", "success")
    
    # Redirect back to the page showing the frames
    return redirect(url_for('retrieve_frames'))

# Route to serve the combined video feed

@app.route('/combined_feed')
def combined_feed():
    return Response(generate_combined_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Flask route to initialize all RTSP streams

# Route to initialize all RTSP streams (triggered when /start_streams is accessed)
@app.route('/start_streams')
def start_streams():
    global streams, capture_threads

    for i, rtsp_url in enumerate(RTSP_URLS, start=1):
        if i not in streams:
            streams[i] = {"latest_frame": None, "lock": Lock(), "active": True}
        if i not in capture_threads or not capture_threads[i].is_alive():
            capture_threads[i] = Thread(target=capture_frames, args=(rtsp_url, i), daemon=True)
            capture_threads[i].start()

    return redirect(url_for('index'))

def initialize_streams():
    global streams, capture_threads
    
    for i, rtsp_url in enumerate(RTSP_URLS, start=1):
        if i not in streams:
            streams[i] = {"latest_frame": None, "lock": Lock(), "active": True}
            capture_threads[i] = Thread(target=capture_frames, args=(rtsp_url, i), daemon=True)
            capture_threads[i].start()
            

# Modified video feed route to show the feed after streams are initialized
@app.route('/video_feed/<int:stream_id>')
def video_feed(stream_id):
    global streams, capture_threads

    if stream_id not in streams or not streams[stream_id]["active"]:
        flash(f"Stream {stream_id} was inactive. Restarting...", "info")
        streams[stream_id] = {"latest_frame": None, "lock": Lock(), "active": True}
        capture_threads[stream_id] = Thread(target=capture_frames, args=(RTSP_URLS[stream_id - 1], stream_id), daemon=True)
        capture_threads[stream_id].start()

    return Response(generate_frames(stream_id), mimetype='multipart/x-mixed-replace; boundary=frame')


# Flask route to stop all streams (optional, for cleanup)
@app.route('/stop_streams')
def stop_streams():
    global streams, capture_threads
    for stream_id in streams.keys():
        streams[stream_id]["active"] = False
        capture_threads[stream_id].join()
    streams.clear()
    capture_threads.clear()
    flash("All streams stopped successfully.", "success")
    return redirect(url_for('index'))
@app.route('/')
def index():
    global streams, capture_threads
   

    if 'user_id' not in session:
        flash('Please log in to access this page.', 'warning')
        return redirect(url_for('login'))
    
    load_rtsp_links(file_path1)
    num_streams = len(RTSP_URLS)
    grid_size = ceil(sqrt(num_streams))

    if not streams or not all(stream_id in streams for stream_id in range(1, num_streams + 1)):
        
        initialize_streams()
        
    

    return render_template('index.html', username=session['username'], num_streams=num_streams, grid_size=grid_size)


# Function to generate a combined feed from all streams


# Existing routes for detected frames, file uploads, email sending, user management, and downloading

# Run Background Processing Thread
processing_thread = Thread(target=process_frames, daemon=True)
processing_thread.start()


@app.route('/detected_frames')
def detected_frames():
    frame_files = sorted(os.listdir(FRAME_SAVE_PATH),
        key=lambda x: os.path.getmtime(os.path.join(FRAME_SAVE_PATH, x)),
        reverse=True )
    return render_template('detected_frames.html', frames=frame_files)



@app.route('/download_frames', methods=['POST'])
def download_frames():
    selected_frames = request.form.getlist('frames')
    zip_name = f"frames_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip"
    zip_path_static = os.path.join(FRAME_SAVE_PATH, zip_name)
    zip_path_arc = os.path.join(ARC_PATH, zip_name)
    with zipfile.ZipFile(zip_path_static, 'w') as zf_static, zipfile.ZipFile(zip_path_arc, 'w') as zf_arc:
        for frame in selected_frames:
            frame_path_static = os.path.join(FRAME_SAVE_PATH, frame)
            frame_path_desktop = os.path.join(FRAMES_PATH, frame)
            if os.path.exists(frame_path_static):
                zf_static.write(frame_path_static, arcname=frame)
            if os.path.exists(frame_path_desktop):
                zf_arc.write(frame_path_desktop, arcname=frame)
    return send_file(zip_path_static, as_attachment=True)



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist('file')
        for file in files:
            if file:
                file_path = os.path.join(FRAME_SAVE_PATH, file.filename)
                file.save(file_path)
                upload_to_dropbox(file_path, file.filename)
                flash(f"File '{file.filename}' uploaded successfully to Dropbox!", 'success')
        return redirect(url_for('upload'))
    return render_template('dropbox.html')



@app.route('/send_email', methods=['POST'])
def send_email():
    recipient = request.form.get('recipient')
    subject = request.form.get('subject') or "Detected Frames"
    body = request.form.get('body') or "Please find the attached frames."
    selected_frames = request.form.getlist('frames')
    if not selected_frames:
        return "No frames selected for email.", 400
    msg = Message(subject, recipients=[recipient], body=body)
    for frame in selected_frames:
        frame_path = os.path.join(FRAME_SAVE_PATH, frame)
        with open(frame_path, 'rb') as f:
            msg.attach(frame, 'image/jpeg', f.read())
    try:
        mail.send(msg)
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {str(e)}", 500



@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            # Initialize streams after login
            load_rtsp_links(file_path1)
            initialize_streams()
            return redirect(url_for('index'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html', form=form)
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except:
            flash('Username or email already exists.', 'danger')
    return render_template('register.html', form=form)



@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/get_frame_metadata/<frame_name>', methods=['GET'])
def get_frame_metadata(frame_name):
    # Query MongoDB for metadata
    metadata = frames_collection.find_one({"frame_name": frame_name}, {"_id": 0})  # Exclude MongoDB ID
    if metadata:
        return jsonify(metadata)
    return jsonify({"error": "Frame not found"}), 404
# Route to retrieve uploaded frames based on metadata
# Route to retrieve uploaded frames based on metadata
@app.route('/retrieve_frames', methods=['GET'])
def retrieve_frames():
    """Retrieve frames with optional start_date and end_date filters."""
    start_date = request.args.get('start_date')  # Expected format: YYYY-MM-DD
    end_date = request.args.get('end_date')  # Expected format: YYYY-MM-DD

    # MongoDB query conditions
    query = {}
    if start_date:
        try:
            start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
            query['upload_time'] = {'$gte': start_date_dt}
        except ValueError:
            return jsonify({'error': 'Invalid start_date format. Use YYYY-MM-DD.'}), 400
    if end_date:
        try:
            end_date_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)  # Include the entire day
            query['upload_time']['$lt'] = end_date_dt
        except ValueError:
            return jsonify({'error': 'Invalid end_date format. Use YYYY-MM-DD.'}), 400

    # Retrieve filtered frames from MongoDB
    metadata = frames_collection.find(query)
    frames = [{"frame_name": doc.get("frame_name"), "file_url": doc.get("file_url")} for doc in metadata]

    # Render the retrieve_frames.html template with the filtered frames
    return render_template('retrieve_frames.html', frames=frames)



@app.route('/retrieve_frames_by_date', methods=['GET'])
def retrieve_frames_by_date():
    """Retrieve frames using a single date filter."""
    date_filter = request.args.get('date')  # Optional date filter in YYYY-MM-DD format
    
    # Query MongoDB for frames uploaded on the specific date or all frames
    if date_filter:
        metadata = frames_collection.find({"upload_time": {"$gte": datetime.strptime(date_filter, '%Y-%m-%d')}})
    else:
        metadata = frames_collection.find()  # Retrieve all frames

    # Prepare the frames list to pass to the template
    frames = []
    for doc in metadata:
        frame_name = doc.get("frame_name")
        file_url = doc.get("file_url")  # Use .get() to avoid KeyError if 'file_url' is missing
        
        # Debugging to print the retrieved fields
        print(f"Retrieved frame: {frame_name}, File URL: {file_url}")
        
        # Only add to the frames list if file_url exists
        if frame_name and file_url:
            frames.append({"frame_name": frame_name, "file_url": file_url})

    # Pass frames to the template
    return render_template('retrieve_frames.html', frames=frames)


@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    if 'file' not in request.files:
        flash('No file uploaded.', 'danger')
        return redirect(url_for('index'))

    file = request.files['file']
    if file:
        # Save the file locally temporarily
        frame_name = secure_filename(file.filename)
        frame_path = os.path.join(FRAME_SAVE_PATH, frame_name)
        file.save(frame_path)

        # Upload the frame to Dropbox and save metadata in MongoDB
        user_id = session.get('user_id', 'anonymous')  # Replace with actual user ID logic
        file_url = upload_frame_to_dropbox_and_mongo(frame_path, frame_name, user_id)

        flash(f"Frame uploaded successfully! URL: {file_url}", 'success')
        return redirect(url_for('index'))

    flash('File upload failed.', 'danger')
    return redirect(url_for('index'))

@app.route("/get_stream_count")
def get_stream_count():
    global streams
    return jsonify({"count": len(streams)})



class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')



class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')




    
if __name__ == "__main__":
    # Run Flask app
    webbrowser.open("http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
