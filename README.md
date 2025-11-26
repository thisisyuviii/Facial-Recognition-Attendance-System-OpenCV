# Face Recognition Attendance System (OpenCV)

![Project Banner](assets/banner.png)

A Python-based **Face Recognition Attendance System** using **OpenCV LBPH** that captures, trains, and recognizes faces in real-time to mark attendance automatically.  
This project works fully offline — no dlib, no internet, no complex libraries.

---

## 📌 Features

### ✔ Face Registration  
- Capture face dataset using a webcam  
- Assign numeric ID + name  
- Saves dataset automatically  

### ✔ Model Training  
- Uses **LBPH algorithm**  
- Saves trained model to `trainer.yml`

### ✔ Real-Time Recognition  
- Detects and identifies faces live  
- Shows name + confidence score  
- Works for multiple people at once  

### ✔ Attendance Logging  
- Saves name, time, date  
- Prevents duplicate entries  

---

# 📸 Screenshots

### 📷 Registration Screen  
![Registration](assets/screenshots/register.png)

### 📷 Training Output  
![Training](assets/screenshots/training.png)

### 📷 Real-Time Recognition  
![Recognition](assets/screenshots/recognition1.png)
![Recognition](assets/screenshots/recognition2.png)

### 📷 Attendance Log  
![Attendance](assets/screenshots/attendance.png)

---

# 📂 Project Structure

```
FaceAttendance/
│
├── main.py
├── dataset/
├── trainer.yml
├── labels.csv
├── Attendance.csv
├── assets/
│   ├── banner.png
│   └── screenshots/
│       ├── register.png
│       ├── training.png
│       ├── recognition.png
│       └── attendance.png
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install opencv-contrib-python numpy pandas
```

### 2️⃣ Run Program

```bash
python main.py
```

Menu options:

```
1 → Register
2 → Train
3 → Start Attendance
```

---

## 🧑‍🏫 Tips for Best Accuracy

- Capture 20–30 images per person  
- Use good lighting  
- Keep face centered  
- Avoid multiple faces during registration  

