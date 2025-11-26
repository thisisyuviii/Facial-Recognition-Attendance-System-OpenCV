import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime, date

# ------------ CONFIG -------------
DATASET_DIR = "dataset"
TRAINER_FILE = "trainer.yml"
LABELS_FILE = "labels.csv"
ATTENDANCE_FILE = "Attendance.csv"
SAMPLES_PER_PERSON = 40  
# ----------------------------------

# Haar cascade for face detection (comes with OpenCV)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def ensure_dirs():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)


def load_labels():
    """Read labels.csv -> dict {id: name}"""
    if not os.path.exists(LABELS_FILE):
        return {}
    df = pd.read_csv(LABELS_FILE)
    return dict(zip(df["id"], df["name"]))


def save_labels(labels_dict):
    df = pd.DataFrame(
        [{"id": int(i), "name": n} for i, n in labels_dict.items()]
    )
    df.to_csv(LABELS_FILE, index=False)


def register_person():
    """Capture face images for a new person."""
    ensure_dirs()
    labels = load_labels()

    # --- get id and name ---
    try:
        person_id = int(input("Enter numeric ID for the person (e.g., 1): "))
    except ValueError:
        print("ID must be a number.")
        return

    if person_id in labels:
        print(f"ID {person_id} already exists for {labels[person_id]}.")
        return

    name = input("Enter name of the person: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    labels[person_id] = name
    save_labels(labels)
    print(f"[INFO] Registered {name} with ID {person_id}. Now capturing faces...")

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            # draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            count += 1
            # crop and save face image
            face_img = gray[y:y + h, x:x + w]
            img_path = os.path.join(
                DATASET_DIR, f"user.{person_id}.{count}.jpg"
            )
            cv2.imwrite(img_path, face_img)
            cv2.putText(
                frame,
                f"Capturing {count}/{SAMPLES_PER_PERSON}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Register - Press 'q' to cancel", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Registration cancelled.")
            break

        if count >= SAMPLES_PER_PERSON:
            print(f"[INFO] Collected {SAMPLES_PER_PERSON} samples for {name}.")
            break

    cap.release()
    cv2.destroyAllWindows()


def train_model():
    """Train LBPH model from images in dataset/."""
    ensure_dirs()
    labels = load_labels()
    if not labels:
        print("[WARN] No labels found. Please register at least one person first.")
        return

    image_paths = [
        os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)
        if f.lower().endswith(".jpg")
    ]

    if not image_paths:
        print("[WARN] No images in dataset/. Please register persons first.")
        return

    face_samples = []
    ids = []

    print(f"[INFO] Found {len(image_paths)} images. Training model...")

    for image_path in image_paths:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        # File name format: user.<id>.<count>.jpg
        try:
            filename = os.path.basename(image_path)
            person_id = int(filename.split(".")[1])
        except Exception:
            continue

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_samples.append(gray[y:y + h, x:x + w])
            ids.append(person_id)

    if not face_samples:
        print("[ERROR] No faces found in dataset images.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_samples, np.array(ids))
    recognizer.write(TRAINER_FILE)
    print(f"[INFO] Training complete. Model saved to {TRAINER_FILE}.")


def ensure_attendance_file():
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["date", "time", "name"])
        df.to_csv(ATTENDANCE_FILE, index=False)


def mark_attendance(name):
    ensure_attendance_file()
    today = date.today().isoformat()
    now = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(ATTENDANCE_FILE)

    # check if already marked today
    if not df[(df["date"] == today) & (df["name"] == name)].empty:
        return  # already marked

    new_row = {"date": today, "time": now, "name": name}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)
    print(f"[ATTENDANCE] {name} marked present at {now}")


def run_attendance():
    """Start webcam, recognize faces, and mark attendance."""
    if not os.path.exists(TRAINER_FILE):
        print("[ERROR] No trained model found. Please run training first.")
        return

    labels = load_labels()
    if not labels:
        print("[ERROR] No labels found.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting attendance. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Could not read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=6, minSize=(120, 120)
        )

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            id_pred, confidence = recognizer.predict(face_img)

            # Improved confidence thresholding
            if confidence < 60:  
                name = labels.get(id_pred, "Unknown")
            else:
                name = "Unknown"

            # draw rectangle + name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name} ({int(confidence)})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if name != "Unknown":
                mark_attendance(name)

        cv2.imshow("Attendance - Press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main_menu():
    while True:
        print("\n===== Face Recognition Attendance (OpenCV) =====")
        print("1. Register new person")
        print("2. Train model")
        print("3. Start attendance")
        print("4. Exit")
        choice = input("Enter choice (1-4): ").strip()

        if choice == "1":
            register_person()
        elif choice == "2":
            train_model()
        elif choice == "3":
            run_attendance()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main_menu()
