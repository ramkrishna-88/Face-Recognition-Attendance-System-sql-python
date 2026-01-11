import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
# --------- Paths ---------
dataset_path = "faces"
os.makedirs(dataset_path, exist_ok=True)

trainer_path = "trainer.yml"
labels_path = "labels.npy"
attendance_folder = "attendance"
os.makedirs(attendance_folder, exist_ok=True)

# --------- Step 1: Capture Faces ---------
def capture_faces():
    uid = uid_entry.get().strip()
    name = name_entry.get().strip()
    
    if uid == "" or name == "":
        messagebox.showwarning("Input Error", "Please enter both UID and Name!")
        return

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            cv2.imwrite(f"{dataset_path}/{uid}{name}{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Capturing Faces - Press 'Q' to Quit", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')] or count >= 50:
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Capture Complete", f"Captured {count} images for {name} (UID: {uid})")
    status_label.config(text=f"✅ {count} images captured for {name} (UID: {uid})")

# --------- Step 2: Train Recognizer ---------
def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_dict = {}  # UID -> numeric ID
    current_id = 0

    for file in os.listdir(dataset_path):
        if file.endswith(".jpg"):
            path = os.path.join(dataset_path, file)
            gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            uid = file.split("_")[0]
            name = file.split("_")[0]
            if uid not in label_dict:
                label_dict[uid] = {"id": current_id, "name": name}
                current_id += 1
            faces.append(gray_img)
            labels.append(label_dict[uid]["id"])

    if not faces:
        messagebox.showwarning("No Data", "No faces found! Capture first.")
        return

    recognizer.train(faces, np.array(labels))
    recognizer.save(trainer_path)
    np.save(labels_path, label_dict)
    messagebox.showinfo("Training", "✅ Training completed successfully!")
    status_label.config(text="✅ Training completed!")

# --------- Step 3: Mark Attendance ---------
def mark_attendance():
    if not os.path.exists(trainer_path):
        messagebox.showwarning("Error", "Train recognizer first!")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(trainer_path)

    label_dict = np.load(labels_path, allow_pickle=True).item()
    id_to_uid_name = {v["id"]: (uid, v["name"]) for uid, v in label_dict.items()}

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    attendance = {}

    messagebox.showinfo("Info", "Press 'Q' to stop attendance window")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            id_, conf = recognizer.predict(face_img)
            if conf < 50:
                uid, name = id_to_uid_name[id_]
                now = datetime.now().strftime("%H:%M:%S")
                attendance[uid] = {"Name": name, "Time": now}
                cv2.putText(frame, f"{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Attendance - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

    if attendance:
        today = datetime.now().strftime("%Y-%m-%d")
        data = [{"UID": uid, "Name": info["Name"], "Time": info["Time"], "Date": today} for uid, info in attendance.items()]
        df = pd.DataFrame(data)
        csv_file = os.path.join(attendance_folder, f"attendance_{today}.csv")
        df.to_csv(csv_file, index=False)
        messagebox.showinfo("Saved", f"✅ Attendance saved: {csv_file}")
        status_label.config(text=f"✅ Attendance saved for {len(attendance)} people")
    else:
        messagebox.showwarning("No Face Detected", "No attendance recorded!")

# --------- Step 4: Check Attendance by UID ---------
def check_attendance():
    uid = check_uid_entry.get().strip()
    today = datetime.now().strftime("%Y-%m-%d")
    csv_file = os.path.join(attendance_folder, f"attendance_{today}.csv")

    if not os.path.exists(csv_file):
        messagebox.showwarning("No Data", "No attendance marked today!")
        return

    history = pd.read_csv(csv_file)
    user_history = history[history["UID"].astype(str) == uid]

    for widget in history_frame.winfo_children():
        widget.destroy()  # Clear previous history

    if not user_history.empty:
        tk.Label(history_frame, text=f"Attendance for UID: {uid}", font=("Helvetica", 12, "bold"), bg="#0D1117", fg="lightgreen").pack()
        for index, row in user_history.iterrows():
            tk.Label(history_frame, text=f"{row['Name']} - {row['Time']}", font=("Helvetica", 12), bg="#0D1117", fg="white").pack()
    else:
        tk.Label(history_frame, text=f"No attendance found for UID: {uid}", font=("Helvetica", 12, "bold"), bg="#0D1117", fg="red").pack()

# --------- Step 5: GUI Setup ---------
root = tk.Tk()
root.title("🎥 Face Recognition Attendance System")
root.geometry("700x600")
root.configure(bg="#0D1117")

# Title
title_frame = tk.Frame(root, bg="#161B22", pady=10)
title_frame.pack(fill="x")
title_label = tk.Label(title_frame, text="Face Recognition Attendance", font=("Helvetica", 20, "bold"), fg="#58FAF4", bg="#161B22")
title_label.pack()

# Input Frame
input_frame = tk.LabelFrame(root, text="👤 Enter Details", font=("Helvetica", 12, "bold"), fg="#FFFFFF", bg="#1E1E2F", padx=20, pady=20)
input_frame.pack(pady=20, padx=20, fill="x")

tk.Label(input_frame, text="UID:", font=("Helvetica", 12), fg="white", bg="#1E1E2F").grid(row=0, column=0, padx=10, pady=10, sticky="w")
uid_entry = tk.Entry(input_frame, font=("Helvetica", 12), width=15)
uid_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")

tk.Label(input_frame, text="Name:", font=("Helvetica", 12), fg="white", bg="#1E1E2F").grid(row=1, column=0, padx=10, pady=10, sticky="w")
name_entry = tk.Entry(input_frame, font=("Helvetica", 12), width=30)
name_entry.grid(row=1, column=1, padx=10, pady=10)

# Buttons Frame
button_frame = tk.Frame(root, bg="#0D1117")
button_frame.pack(pady=10)

def create_button(text, command, bg_color, fg_color):
    btn = tk.Button(button_frame, text=text, command=command, bg=bg_color, fg=fg_color, font=("Helvetica", 12, "bold"), activebackground="#FFFFFF", activeforeground=bg_color, width=20, relief="ridge", bd=3)
    return btn

btn1 = create_button("📸 Capture Faces", capture_faces, "#0078D7", "white")
btn2 = create_button("🧠 Train Recognizer", train_recognizer, "#28A745", "white")
btn3 = create_button("🕵 Mark Attendance", mark_attendance, "#FFC107", "black")
btn1.grid(row=0, column=0, padx=15, pady=10)
btn2.grid(row=0, column=1, padx=15, pady=10)
btn3.grid(row=1, column=0, padx=15, pady=10)

# Check Attendance Frame
check_frame = tk.LabelFrame(root, text="🔍 Check Attendance", font=("Helvetica", 12, "bold"), fg="white", bg="#1E1E2F", padx=20, pady=20)
check_frame.pack(pady=10, padx=20, fill="x")

tk.Label(check_frame, text="Enter UID:", font=("Helvetica", 12), fg="white", bg="#1E1E2F").grid(row=0, column=0, padx=10, pady=10, sticky="w")
check_uid_entry = tk.Entry(check_frame, font=("Helvetica", 12), width=15)
check_uid_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")

btn_check = tk.Button(check_frame, text="✅ Show Attendance", command=check_attendance, font=("Helvetica", 12, "bold"), bg="#17A2B8", fg="white", width=20)
btn_check.grid(row=0, column=2, padx=10, pady=10)

history_frame = tk.Frame(root, bg="#0D1117")
history_frame.pack(pady=10, padx=20, fill="both", expand=True)

# Status Bar
status_label = tk.Label(root, text="Ready", font=("Helvetica", 12), fg="lightgreen", bg="#0D1117")
status_label.pack(side="bottom", fill="x", pady=10)

root.mainloop()