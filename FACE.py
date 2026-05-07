import cv2
import face_recognition
import pickle
import mediapipe as mp
import pyautogui
import tkinter as tk
import os
import time

pyautogui.FAILSAFE = False
SCREEN_W, SCREEN_H = pyautogui.size()

print("Loading your face data...")
with open("encodings.pkl", "rb") as f:
    known_encodings = pickle.load(f)
print(f"Loaded {len(known_encodings)} face encodings. Ready!")

mp_hands = mp.solutions.hands
hands_tracker = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

vault_open = False
vault_win  = None

def open_vault():
    global vault_win, vault_open
    if vault_open:
        return
    vault_open = True
    vault_win = tk.Tk()
    vault_win.title("VAULT UNLOCKED")
    vault_win.geometry("420x320")
    vault_win.configure(bg="#0d1117")
    vault_win.resizable(False, False)
    tk.Label(vault_win,
             text="  VAULT UNLOCKED  ",
             font=("Courier", 18, "bold"),
             fg="#00ff88", bg="#0d1117").pack(pady=18)
    tk.Label(vault_win,
             text="Your files:",
             font=("Courier", 11),
             fg="#888888", bg="#0d1117").pack(anchor="w", padx=24)
    files = os.listdir("vault_files")
    if files:
        for fname in files:
            tk.Label(vault_win,
                     text=f"   {fname}",
                     font=("Courier", 13),
                     fg="#ffffff", bg="#0d1117").pack(anchor="w", padx=24, pady=2)
    else:
        tk.Label(vault_win,
                 text="   (no files yet)",
                 font=("Courier", 12),
                 fg="#555555", bg="#0d1117").pack(anchor="w", padx=24)
    tk.Label(vault_win,
             text="Make a FIST to close vault",
             font=("Courier", 10),
             fg="#444444", bg="#0d1117").pack(side="bottom", pady=14)
    vault_win.update()

def close_vault():
    global vault_win, vault_open
    if vault_win:
        try:
            vault_win.destroy()
        except Exception:
            pass
    vault_win  = None
    vault_open = False

def finger_distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

# FIX 1: lower resolution = faster
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

last_face_time = 0
clicked        = False

print("Running! Look at the camera to unlock.")
print("Finger = move mouse  |  Pinch = click  |  Fist = close vault")
print("Press Q to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- GESTURE MOUSE ---
    hand_result = hands_tracker.process(rgb)
    if hand_result.multi_hand_landmarks:
        lm = hand_result.multi_hand_landmarks[0].landmark

        mouse_x = int(lm[8].x * SCREEN_W)
        mouse_y = int(lm[8].y * SCREEN_H)
        pyautogui.moveTo(mouse_x, mouse_y, duration=0.02)

        dist = finger_distance(lm[4], lm[8])
        if dist < 0.05 and not clicked:
            pyautogui.click()
            clicked = True
        elif dist >= 0.05:
            clicked = False

        tips     = [8, 12, 16, 20]
        knuckles = [6, 10, 14, 18]
        all_curled = all(lm[t].y > lm[k].y for t, k in zip(tips, knuckles))
        if all_curled:
            close_vault()

        dot_x = int(lm[8].x * frame.shape[1])
        dot_y = int(lm[8].y * frame.shape[0])
        cv2.circle(frame, (dot_x, dot_y), 8, (0, 255, 0), -1)
        cv2.circle(frame, (dot_x, dot_y), 10, (255, 255, 255), 2)

    # FIX 2: check face every 2 seconds instead of 1
    now = time.time()
    if now - last_face_time >= 2.0:
        last_face_time = now

        # FIX 3: use HOG model - much faster on Pi 4 CPU
        face_locations = face_recognition.face_locations(rgb, model="hog")
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        my_face_found = False
        for enc in face_encodings:
            matches = face_recognition.compare_faces(
                known_encodings, enc, tolerance=0.55)
            if True in matches:
                my_face_found = True
                break

        if my_face_found:
            open_vault()
        else:
            close_vault()

    # --- STATUS ON SCREEN ---
    if vault_open:
        cv2.rectangle(frame, (0, 0), (310, 45), (0, 60, 0), -1)
        cv2.putText(frame, "VAULT: UNLOCKED", (8, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)
    else:
        cv2.rectangle(frame, (0, 0), (270, 45), (60, 0, 0), -1)
        cv2.putText(frame, "VAULT: LOCKED", (8, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 2)

    cv2.putText(frame, "Q=quit", (270, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Scale up display so it looks bigger on screen
    display = cv2.resize(frame, (640, 480))
    cv2.imshow("Face Vault - Raspberry Pi", display)

    if vault_win:
        try:
            vault_win.update()
        except Exception:
            vault_open = False
            vault_win  = None

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

close_vault()
cam.release()
cv2.destroyAllWindows()
print("Goodbye!")