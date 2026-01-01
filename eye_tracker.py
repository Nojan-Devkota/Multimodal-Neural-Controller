import cv2
import mediapipe as mp
import math
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
GESTURE_MODEL_PATH = 'gesture_recognizer.task'
FACE_MODEL_PATH = 'face_landmarker.task'

WINDOW_NAME = "Final Eye & Hand Control"
WIDTH, HEIGHT = 1280, 720
TOGGLE_BTN_AREA = ((20, 20), (220, 80))
CALIB_BTN_AREA  = ((240, 20), (440, 80))

# --- APP STATE ---
class Calibration:
    def __init__(self):
        self.center_val = 0.5
        self.left_val = 0.4
        self.right_val = 0.6
        self.is_calibrated = False

class AppState:
    def __init__(self):
        self.x, self.y = WIDTH // 2, HEIGHT // 2
        self.radius = 60
        self.colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
        self.color_idx = 0
        self.current_color = self.colors[0]
        
        self.mode = "HAND" # "HAND", "EYE", "CALIBRATING"
        self.prev_hand_angle = None
        self.last_fist_time = 0
        self.last_blink_time = 0
        
        # Smoothing variables
        self.smoothed_gaze = 0.5
        self.alpha = 0.15    # Smoothing factor
        self.deadzone = 0.05 # Noise threshold
        
        self.calib = Calibration()

state = AppState()

# --- MATHEMATICAL HELPER FUNCTIONS ---

def get_dist(p1, p2):
    """
    Calculates Euclidean distance between two 2D points.
    Formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)
    Used to measure eye opening height and width.
    """
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def get_hand_rotation(wrist, knuckle):
    """
    Calculates the angle of the hand in 2D space.
    Math: 
    1. We treat the hand as a vector from Wrist(x1,y1) to Knuckle(x2,y2).
    2. dy = y2 - y1, dx = x2 - x1
    3. math.atan2(dy, dx) returns the angle in radians (-PI to PI).
    4. We convert radians to degrees for easier logic.
    """
    return math.degrees(math.atan2(knuckle.y - wrist.y, knuckle.x - wrist.x))

def get_iris_position(face_landmarks):
    """
    Calculates the horizontal position of the iris relative to the eye corners.
    Math:
    We perform a Linear Interpolation (Inverse Lerp).
    Ratio = (Iris_X - Outer_Corner_X) / (Inner_Corner_X - Outer_Corner_X)
    
    Returns:
    0.0 -> Looking completely away (towards ear)
    1.0 -> Looking completely inward (towards nose)
    """
    # Indices for Left Eye (appearing on Left in mirrored view)
    p_outer = face_landmarks[33]  # Outer Corner
    p_inner = face_landmarks[133] # Inner Corner (Near nose)
    p_iris  = face_landmarks[468] # Center of Iris
    
    eye_width = p_inner.x - p_outer.x
    if eye_width == 0: return 0.5
    
    dist_to_iris = p_iris.x - p_outer.x
    return dist_to_iris / eye_width

# --- INITIALIZATION ---
base_hand = python.BaseOptions(model_asset_path=GESTURE_MODEL_PATH)
opt_hand = vision.GestureRecognizerOptions(base_options=base_hand)
recognizer = vision.GestureRecognizer.create_from_options(opt_hand)

base_face = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
opt_face = vision.FaceLandmarkerOptions(
    base_options=base_face,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True, 
    num_faces=1)
face_landmarker = vision.FaceLandmarker.create_from_options(opt_face)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    hand_result = recognizer.recognize(mp_image)
    face_result = face_landmarker.detect(mp_image)
    
    # -----------------------------
    # 1. FACE & EYE LOGIC
    # -----------------------------
    current_raw_gaze = 0.5 
    is_blinking = False
    
    if face_result.face_landmarks:
        fl = face_result.face_landmarks[0]
        
        # Draw Iris (Yellow) for visual confirmation
        for idx in range(468, 473):
            pt = fl[idx]
            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0, 255, 255), -1)
        
        # --- BLINK DETECTION MATH ---
        # We calculate the Aspect Ratio of the eye.
        # Ratio = Height / Width.
        # Normal open eye is ~0.25 - 0.35. A closed eye drops below 0.20.
        eye_h = get_dist(fl[159], fl[145])
        eye_w = get_dist(fl[33], fl[133])
        blink_ratio = eye_h / (eye_w + 1e-6)
        
        is_blinking = blink_ratio < 0.22 
        
        # Action: Change Color (ONLY IN EYE MODE)
        if is_blinking and state.mode == "EYE":
            cv2.putText(frame, "BLINK DETECTED", (w//2 - 60, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            if time.time() - state.last_blink_time > 0.5:
                state.color_idx = (state.color_idx + 1) % len(state.colors)
                state.current_color = state.colors[state.color_idx]
                state.last_blink_time = time.time()
                
        # --- GAZE STABILIZATION MATH ---
        current_raw_gaze = get_iris_position(fl)
        
        if not is_blinking:
            # Low-Pass Filter (Exponential Moving Average)
            # New_Val = (Old_Val * 0.85) + (New_Input * 0.15)
            # This smooths out high-frequency jitter from the camera sensor.
            state.smoothed_gaze = (state.smoothed_gaze * (1 - state.alpha)) + (current_raw_gaze * state.alpha)
        
        # --- GAZE MOVEMENT MATH ---
        if state.mode == "EYE" and state.calib.is_calibrated:
            
            # Calculate deviation from the User's specific calibrated Center
            diff_from_center = state.smoothed_gaze - state.calib.center_val
            move_x = state.x 
            
            # Deadzone Check: Ignore movements smaller than threshold (0.05)
            # This prevents the ball from shaking when looking straight.
            if abs(diff_from_center) > state.deadzone:
                
                # Math: Normalize the gaze intensity based on calibration range.
                # Intensity = (Current_Diff - Deadzone) / (Max_Range)
                # This ensures speed starts at 0 and ramps up to 1.0 (Full Speed).
                
                if diff_from_center < 0: # Looking Left
                    dist = abs(diff_from_center) - state.deadzone
                    max_range = abs(state.calib.center_val - state.calib.left_val) + 1e-6
                    intensity = min(1.0, dist / max_range)
                    target_pos = (w // 2) - (intensity * (w // 2))
                    
                else: # Looking Right
                    dist = abs(diff_from_center) - state.deadzone
                    max_range = abs(state.calib.right_val - state.calib.center_val) + 1e-6
                    intensity = min(1.0, dist / max_range)
                    target_pos = (w // 2) + (intensity * (w // 2))
                
                # Linear Interpolation for smooth screen movement
                state.x = int(state.x * 0.1 + target_pos * 0.9)
                state.y = int(state.y * 0.1 + (h//2) * 0.9)

    # -----------------------------
    # 2. HAND LOGIC
    # -----------------------------
    if hand_result.gestures:
        gesture = hand_result.gestures[0][0].category_name
        landmarks = hand_result.hand_landmarks[0]
        
        wrist = landmarks[0]
        mid_mcp = landmarks[9] # Middle finger knuckle (stable pivot point)
        idx_tip = landmarks[8]
        
        # A. Hand Move (Only in HAND MODE)
        if state.mode == "HAND" and (gesture == "Pointing_Up" or gesture == "Victory"):
            state.x = int(idx_tip.x * w)
            state.y = int(idx_tip.y * h)
            
        # B. Hand Resize (Works in BOTH modes)
        # We detect rotation of the wrist relative to the hand's center line.
        if gesture == "Open_Palm":
            angle = get_hand_rotation(wrist, mid_mcp)
            
            if state.prev_hand_angle is not None:
                # Calculate delta (change in angle)
                delta = angle - state.prev_hand_angle
                
                # Handle angle wrapping at 180/-180 degrees
                if delta > 180: delta -= 360
                elif delta < -180: delta += 360
                
                # Apply change to radius
                if abs(delta) > 1.0: # Filter small jitters
                    state.radius += delta * 2.0
                    state.radius = max(20, min(300, state.radius))
                    
            state.prev_hand_angle = angle
            cv2.putText(frame, f"Rotation: {int(angle)}", (int(wrist.x*w), int(wrist.y*h)+40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        else:
            state.prev_hand_angle = None
            
        # C. Hand Color (Fist) - Works in Hand Mode
        if gesture == "Closed_Fist" and state.mode == "HAND":
             if time.time() - state.last_fist_time > 1.0:
                state.color_idx = (state.color_idx + 1) % len(state.colors)
                state.current_color = state.colors[state.color_idx]
                state.last_fist_time = time.time()

    # -----------------------------
    # 3. DRAW UI
    # -----------------------------
    
    cv2.circle(frame, (int(state.x), int(state.y)), int(state.radius), state.current_color, -1)
    
    # Toggle Button
    btn_color = (0, 150, 0) if state.mode == "HAND" else (0, 0, 150)
    cv2.rectangle(frame, TOGGLE_BTN_AREA[0], TOGGLE_BTN_AREA[1], btn_color, -1)
    cv2.putText(frame, f"MODE: {state.mode}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # Calibration Button
    cv2.rectangle(frame, CALIB_BTN_AREA[0], CALIB_BTN_AREA[1], (100, 100, 100), -1)
    cv2.putText(frame, "CALIBRATE", (250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # Calibration Overlay
    if state.mode == "CALIBRATING":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (WIDTH, HEIGHT), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        cv2.putText(frame, "CALIBRATION MODE", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, f"Raw Gaze: {current_raw_gaze:.3f}", (400, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, "1. Look CENTER -> Press 'c'", (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "2. Look LEFT   -> Press 'l'", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "3. Look RIGHT  -> Press 'r'", (300, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "4. Press 'd' when Done",       (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        vals = f"L={state.calib.left_val:.2f} | C={state.calib.center_val:.2f} | R={state.calib.right_val:.2f}"
        cv2.putText(frame, vals, (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow(WINDOW_NAME, frame)
    
    # -----------------------------
    # 4. CONTROLS
    # -----------------------------
    key = cv2.waitKey(1)
    if key & 0xFF == 27: break 
    
    if state.mode == "CALIBRATING":
        if key == ord('c'): state.calib.center_val = current_raw_gaze
        if key == ord('l'): state.calib.left_val = current_raw_gaze
        if key == ord('r'): state.calib.right_val = current_raw_gaze
        if key == ord('d'): 
             state.calib.is_calibrated = True
             state.mode = "EYE"
             print("Calibration Saved.")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if (TOGGLE_BTN_AREA[0][0] < x < TOGGLE_BTN_AREA[1][0] and 
                TOGGLE_BTN_AREA[0][1] < y < TOGGLE_BTN_AREA[1][1]):
                state.mode = "EYE" if state.mode == "HAND" else "HAND"
            
            if (CALIB_BTN_AREA[0][0] < x < CALIB_BTN_AREA[1][0] and 
                CALIB_BTN_AREA[0][1] < y < CALIB_BTN_AREA[1][1]):
                state.mode = "CALIBRATING"

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

cap.release()
cv2.destroyAllWindows()