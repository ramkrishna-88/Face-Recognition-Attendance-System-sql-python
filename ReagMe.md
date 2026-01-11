# Face Recognition Attendance System

A **Python-based Face Recognition Attendance System** with a modern **Tkinter GUI**.  
This project allows capturing face images, training a recognizer, marking attendance using live webcam feed, and saving attendance records automatically in CSV files.

---
## Table of Contents

- <a href="#Project Features">Features</a>
- <a href="#Technologies Used">Technologies Used</a>
- <a href="#Project Structure">Project Structure</a>
- <a href="#Install Required Libraries">Install Required Libraries</a>
- <a href="#How to Run ">How to Run </a>
- <a href="#How to Use">How to Use</a>
- <a href="#Face Recognition Algorithm">Face Recognition Algorithm</a>
- <a href="#Requirements">Requirements</a>
- <a href="#Future Enhancements">Future Enhancements</a>
- <a href="#Author">Author</a>

---
<h2><a class="anchor" id="Project Features"></a>Project Features</h2>

✅ Capture face images using webcam  
✅ Train Face Recognizer (LBPH Algorithm)  
✅ Real-time face recognition  
✅ Automatic attendance marking with date & time  
✅ Attendance saved in CSV files  
✅ Check attendance by UID  
✅ Interactive Tkinter GUI  

---
<h2><a class="anchor" id="Technologies Used"></a>Technologies Used</h2>

- Python 3  
- OpenCV (cv2)  
- NumPy  
- Pandas  
- Tkinter (GUI)  
- Haar Cascade Classifier  
- LBPH Face Recognizer  

---
<h2><a class="anchor" id="Project Structure"></a>Project Structure</h2>

face-recognition-attendance/
│
├── faces/ # Stored face images dataset
├── attendance/ # Saved attendance CSV files
├── trainer.yml # Trained face recognition model
├── labels.npy # UID ↔ Name mapping
├── main.py # Main application code
├── README.md # Project documentation

---
<h2><a class="anchor" id="Install Required Libraries"></a>Install Required Libraries</h2>

pip install opencv-python opencv-contrib-python numpy pandas
Tkinter comes pre-installed with Python.

---
<h2><a class="anchor" id="How to Run"></a>How to Run</h2>

python main.py

---
<h2><a class="anchor" id="How to Use"></a>How to Use</h2>

- Capture Faces
 - Enter UID and Name
 - Click 📸 Capture Faces
 - System captures 50 face images automatically

- Train Recognizer
 - Click 🧠 Train Recognizer
 - Model is trained and saved

- Mark Attendance
 - Click 🕵 Mark Attendance
 - Webcam opens for real-time recognition
 - Recognized faces are recorded with time

- Check Attendance
 - Enter UID
 - Click Show Attendance
 - Displays attendance for the current day

---
<h2><a class="anchor" id="Face Recognition Algorithm"></a>Face Recognition Algorithm</h2>

- Uses LBPH (Local Binary Pattern Histogram)
- Robust for real-time recognition
- Works well under varying lighting conditions

---
<h2><a class="anchor" id="Requirements"></a>Requirements</h2>

- Webcam
- Python 3.7+
- OpenCV-contrib installed

---
<h2><a class="anchor" id="Future Enhancements"></a>Future Enhancements</h2>

- Add multiple-day attendance history
- Export attendance to Excel
- Improve recognition accuracy with Deep Learning
- Add database support (MySQL)

---
## Author

Ram Krishna
Email: ramkrishna000888@gmail.com
Linkeddin: https://www.linkedin.com/in/ramkrishna000/