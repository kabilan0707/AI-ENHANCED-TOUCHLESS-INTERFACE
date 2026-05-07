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
print(f"Loaded {len(known_encodings)} encodings. Ready!")

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
    tk.Label(vault_win, text="  VAULT UNLOCKED  ",
             font=("Courier", 18, "bold"),
             fg="#00ff88", bg="#0d1117").pack(pady=18)
    tk.Label(vault_win, text="Your files:",
             font=("Courier", 11),
             fg="#888888", bg="#0d1117").pack(anchor="w", padx=24)
    files = os.listdir("vault_files")
    if files:
        for fname in files:
            tk.Label(vault_win, text=f"   {fname}",
                     font=("Courier", 13),
                     fg="#ffffff", bg="#0d1117").pack(anchor="w", padx=24, pady=2)
    tk.Label(vault_win, text="Make a FIST to close vault",
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

def dist(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5

def is_pinch(lm):
    return dist(lm[4], lm[8]) < 0.08

def is_fist(lm):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    fingers_curled = all(lm[t].y > lm[p].y for t, p in zip(tips, pips))
    thumb_curled = dist(lm[4], lm[5]) < 0.1
    return fingers_curled and thumb_curled

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

last_face_time = 0
click_cooldown = 0

print("Running! Look at camera to unlock.")
print("Finger=move  |  Pinch=click  |  Fist=close vault  |  Q=quit")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gesture_label = ""

    hand_result = hands_tracker.process(rgb)
    if hand_result.multi_hand_landmarks:
        lm = hand_result.multi_hand_landmarks[0].landmark

        mouse_x = int(lm[8].x * SCREEN_W)
        mouse_y = int(lm[8].y * SCREEN_H)
        pyautogui.moveTo(mouse_x, mouse_y, duration=0.02)
        gesture_label = "pointing"

        if is_fist(lm):
            gesture_label = "FIST - closing vault"
            close_vault()
        elif is_pinch(lm):
            gesture_label = "PINCH - click!"
            if click_cooldown == 0:
                pyautogui.click()
                click_cooldown = 15

        if click_cooldown > 0:
            click_cooldown -= 1

        dot_x = int(lm[8].x * frame.shape[1])
        dot_y = int(lm[8].y * frame.shape[0])
        cv2.circle(frame, (dot_x, dot_y), 8, (0, 255, 0), -1)
        cv2.circle(frame, (dot_x, dot_y), 10, (255, 255, 255), 2)

    now = time.time()
    if now - last_face_time >= 2.0:
        last_face_time = now
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

    if vault_open:
        cv2.rectangle(frame, (0, 0), (310, 45), (0, 60, 0), -1)
        cv2.putText(frame, "VAULT: UNLOCKED", (8, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)
    else:
        cv2.rectangle(frame, (0, 0), (270, 45), (60, 0, 0), -1)
        cv2.putText(frame, "VAULT: LOCKED", (8, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 2)

    if gesture_label:
        cv2.putText(frame, gesture_label, (8, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)

    cv2.putText(frame, "Q=quit", (270, 232),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

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