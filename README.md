Face Recognition Web Application
Introduction

This project is a web application for face recognition, developed using Flask for the backend and a simple HTML/CSS frontend. The application captures video from the webcam, recognizes known faces, and displays the results in real-time. If a new face is detected, it saves the face and its encoding for future recognition. Additionally, it can open a text file associated with each recognized face.
Features

    Real-time Face Recognition: Capture and recognize faces in real-time using a webcam.
    New Face Detection: Automatically saves new faces and their encodings for future recognition.
    Text File Association: Opens a text file associated with each recognized face.
    Web Interface: Start and stop the recognition process through a simple web interface.

Requirements

    Python 3.7+
    Flask
    OpenCV
    face_recognition
    numpy
    PIL (Pillow)
    customtkinter
    torch
    diffusers

    Access the Web Application:
         Open your web browser and navigate to http://0.0.0.0:5000.

    Start Recognition:
        Click the "Start Recognition" button to begin the face recognition process.
        The webcam feed will be displayed, and faces will be recognized in real-time.

    Stop Recognition:
        Click the "Stop Recognition" button to stop the face recognition process.

File Structure

    app.py: Main application file that sets up the Flask server and handles the face recognition logic.
    authtoken.py: Contains the authentication token for accessing external APIs (if needed).
    templates/index.html: HTML template for the web interface.
    faces/: Directory to store known faces and their encodings.
    requirements.txt: List of required Python packages.

Adding New Faces

When a new face is detected, it is automatically saved in the faces directory with a unique name. A text file (info.txt) is created in the same directory, which can be opened to add any relevant information about the face.
Known Issues

    Webcam Access: Ensure your browser has permission to access the webcam.
    Face Recognition Accuracy: The accuracy of face recognition depends on the quality of the webcam and the lighting conditions.
