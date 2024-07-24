from flask import Flask, jsonify, render_template, Response
import cv2
import pathlib
import face_recognition
import numpy as np
import time
import pickle
import os

app = Flask(__name__)

# Ensure the 'faces' directory exists
faces_dir = pathlib.Path('faces')
faces_dir.mkdir(exist_ok=True)

# Path for the file storing known faces
encodings_file_path = faces_dir / "known_faces.pkl"

# Global variables for webcam control
webcam_active = False
camera = None

# Function to load known faces
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    if encodings_file_path.exists():
        with open(encodings_file_path, 'rb') as file:
            data = pickle.load(file)
            known_face_encodings = data['encodings']
            known_face_names = data['names']
    return known_face_encodings, known_face_names

# Save known faces
def save_known_faces(known_face_encodings, known_face_names):
    with open(encodings_file_path, 'wb') as file:
        pickle.dump({'encodings': known_face_encodings, 'names': known_face_names}, file)

# Function to open text file
def open_text_file(file_path):
    if not os.path.isfile(file_path):
        return
    try:
        if os.name == 'posix':  # For Unix-like systems (Linux, macOS)
            os.system(f'xdg-open "{file_path}"')
        elif os.name == 'nt':  # For Windows
            os.startfile(file_path)  # Use os.startfile for Windows
        else:
            print("Unsupported OS for file opening.")
    except Exception as e:
        print(f"Error opening file: {e}")

def generate_frames():
    known_face_encodings, known_face_names = load_known_faces()
    global camera
    global webcam_active

    face_count = len([d for d in faces_dir.iterdir() if d.is_dir()])
    detection_times = {}
    opened_files = set()

    while webcam_active:
        success, frame = camera.read()
        if not success:
            print("Failed to grab frame")
            break
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                color = (0, 255, 0)
                label = "New face"

                if known_face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                    if face_distances.size > 0:
                        best_match_index = np.argmin(face_distances)

                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                            color = (0, 0, 255)
                            label = f"Recognized: {name}"
                            file_path = faces_dir / name / 'info.txt'
                            if name not in opened_files:
                                open_text_file(file_path)
                                opened_files.add(name)

                if name == "Unknown":
                    face_count += 1
                    new_face_dir = faces_dir / f"face_{face_count}"
                    new_face_dir.mkdir(exist_ok=True)
                    text_file_path = new_face_dir / "info.txt"
                    text_file_path.touch()

                    face_image = frame[top:bottom, left:right]
                    face_image_resized = cv2.resize(face_image, (800, 800))
                    new_face_path = new_face_dir / f"{len(list(new_face_dir.iterdir()))}.jpg"
                    cv2.imwrite(str(new_face_path), face_image_resized)
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(new_face_dir.name)
                    detection_times[new_face_dir.name] = time.time()
                    label = f"New face: {new_face_dir.name}"

                    save_known_faces(known_face_encodings, known_face_names)
                    open_text_file(text_file_path)
                    opened_files.add(new_face_dir.name)

                if name != "Unknown":
                    current_time = time.time()
                    for known_name in detection_times.keys():
                        if detection_times[known_name] is not None:
                            elapsed_time = current_time - detection_times[known_name]
                            if elapsed_time > 400:
                                color = (0, 0, 255)
                                label = f"Recognized: {known_name}"
                                detection_times[known_name] = None

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    if camera is not None:
        camera.release()
        camera = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not webcam_active:
        return "Webcam not active", 400
    # Add cache-busting parameter to force reload
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    global webcam_active
    global camera

    if webcam_active:
        return jsonify({"status": "Recognition process already running"}), 400

    webcam_active = True

    # Ensure the previous camera instance is released
    if camera is not None:
        camera.release()
    
    # Initialize a new camera instance
    camera = cv2.VideoCapture(0)
    
    return jsonify({"status": "Recognition process started"}), 200

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global webcam_active
    global camera

    if not webcam_active:
        return jsonify({"status": "Recognition process not running"}), 400

    webcam_active = False

    # Wait a moment to ensure frames stop being sent
    time.sleep(1)  # Adjust the sleep time if necessary

    if camera is not None:
        camera.release()
        camera = None

    return jsonify({"status": "Recognition process stopped"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
