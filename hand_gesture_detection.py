import cv2
import mediapipe as mp
import pyttsx3
import time

# =============== Initialization ===============
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()

engine.setProperty('rate', 160)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # optional: female voice

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

last_spoken = ""
last_time = 0

def speak(text):
    global last_spoken, last_time
    if text != last_spoken or time.time() - last_time > 5:
        engine.say(text)
        engine.runAndWait()
        last_spoken = text
        last_time = time.time()

# ==========================================================
#  Detect gesture or sign language from hand landmarks
# ==========================================================
def detect_sign_language(landmarks):
    """
    Detects ASL signs (A–E) based on finger positions.
    """
    # Finger tip and MCP (base) landmarks
    tips = [4, 8, 12, 16, 20]
    base = [2, 5, 9, 13, 17]

    fingers = []
    for i in range(1, 5):
        if landmarks[tips[i]].y < landmarks[base[i]].y:
            fingers.append(1)  # finger up
        else:
            fingers.append(0)

    thumb_open = landmarks[tips[0]].x > landmarks[base[0]].x

    # --- Detect common ASL letters ---
    if fingers == [0, 0, 0, 0] and not thumb_open:
        return "A"
    elif fingers == [1, 1, 1, 1] and not thumb_open:
        return "B"
    elif fingers == [1, 1, 0, 0] and thumb_open:
        return "C"
    elif fingers == [1, 1, 1, 0] and thumb_open:
        return "D"
    elif fingers == [0, 0, 0, 0] and thumb_open:
        return "E"
    else:
        return None

def detect_gesture(landmarks):
    thumb_tip = landmarks[4].y
    index_tip = landmarks[8].y
    middle_tip = landmarks[12].y
    ring_tip = landmarks[16].y
    pinky_tip = landmarks[20].y

    if thumb_tip < index_tip and index_tip < middle_tip:
        return "👍 Thumbs Up"
    elif thumb_tip > index_tip and index_tip > middle_tip:
        return "👎 Thumbs Down"
    elif index_tip < ring_tip and middle_tip < ring_tip:
        return "✌️ Peace"
    elif abs(index_tip - middle_tip) < 0.02:
        return "🤚 Stop / Hello"
    else:
        return None

# ==========================================================
#                     MAIN LOOP
# ==========================================================
while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    h, w, _ = image.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
            )

            gesture = detect_gesture(hand_landmarks.landmark)
            sign = detect_sign_language(hand_landmarks.landmark)

            detected = sign if sign else (gesture if gesture else "🤚 Hand Detected")

            # Display detected sign
            cv2.rectangle(image, (0, h - 80), (w, h), (0, 0, 0), -1)
            cv2.putText(image, f"Detected: {detected}", (30, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            speak(detected)

    else:
        cv2.rectangle(image, (0, h - 80), (w, h), (50, 50, 50), -1)
        cv2.putText(image, "No Hand Detected", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("🤟 Smart Sign Language & Gesture Detector", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
