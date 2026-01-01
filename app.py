import cv2
import av
import mediapipe as mp
import math
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Eye & Hand Control", layout="wide")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at top left, #1a1a2e, #16213e, #0f3460); }
    h1, h2, h3, p { color: white !important; font-family: 'Inter', sans-serif; }
    .stButton>button {
        background: rgba(255, 255, 255, 0.1); color: white;
        border: 1px solid rgba(255,255,255,0.2); border-radius: 8px;
    }
    .stButton>button:hover {
        border-color: #00ff88; color: #00ff88;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CONFIG ---
GESTURE_MODEL = 'gesture_recognizer.task'
FACE_MODEL = 'face_landmarker.task'

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def get_hand_rotation(wrist, knuckle):
    return math.degrees(math.atan2(knuckle.y - wrist.y, knuckle.x - wrist.x))

def get_iris_position(face_landmarks):
    # Left Eye (Mirrored): 33(Outer), 133(Inner), 468(Iris)
    p_outer = face_landmarks[33]
    p_inner = face_landmarks[133]
    p_iris  = face_landmarks[468]
    eye_width = p_inner.x - p_outer.x
    if eye_width == 0: return 0.5
    return (p_iris.x - p_outer.x) / eye_width

# --- 3. VIDEO PROCESSOR ---
class EyeHandProcessor(VideoTransformerBase):
    def __init__(self):
        # Load AI Models
        try:
            base_hand = python.BaseOptions(model_asset_path=GESTURE_MODEL)
            self.recognizer = vision.GestureRecognizer.create_from_options(
                vision.GestureRecognizerOptions(base_options=base_hand))
            
            base_face = python.BaseOptions(model_asset_path=FACE_MODEL)
            self.face_landmarker = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(
                    base_options=base_face, 
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    num_faces=1))
        except Exception as e:
            print(f"Error loading models: {e}")

        # State
        self.x, self.y = 640, 360
        self.radius = 60
        self.colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
        self.color_idx = 0
        
        # Control Variables (Updated by Streamlit UI)
        self.mode = "HAND"
        self.calib_left = 0.4
        self.calib_center = 0.5
        self.calib_right = 0.6
        self.is_calibrated = False
        
        # Internal Logic
        self.prev_hand_angle = None
        self.last_fist_time = 0
        self.last_blink_time = 0
        self.smoothed_gaze = 0.5
        self.last_raw_gaze = 0.5 # For UI display
        self.alpha = 0.15
        self.deadzone = 0.05

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror
        h, w, _ = img.shape
        
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Run AI
        hand_res = self.recognizer.recognize(mp_img)
        face_res = self.face_landmarker.detect(mp_img)
        
        # --- FACE LOGIC ---
        is_blinking = False
        
        if face_res.face_landmarks:
            fl = face_res.face_landmarks[0]
            
            # 1. Draw Iris (Visual Feedback)
            pt = fl[468]
            cv2.circle(img, (int(pt.x * w), int(pt.y * h)), 4, (0, 255, 255), -1)

            # 2. Blink Detection
            eye_h = get_dist(fl[159], fl[145])
            eye_w = get_dist(fl[33], fl[133])
            if (eye_h / (eye_w + 1e-6)) < 0.22:
                is_blinking = True
                if self.mode == "EYE" and (time.time() - self.last_blink_time > 0.5):
                    self.color_idx = (self.color_idx + 1) % len(self.colors)
                    self.last_blink_time = time.time()
            
            # 3. Gaze Calc
            self.last_raw_gaze = get_iris_position(fl)
            
            # Debug Text on Screen
            cv2.putText(img, f"Gaze: {self.last_raw_gaze:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if not is_blinking:
                self.smoothed_gaze = (self.smoothed_gaze * (1 - self.alpha)) + (self.last_raw_gaze * self.alpha)
            
            # 4. Move Logic
            if self.mode == "EYE" and self.is_calibrated:
                diff = self.smoothed_gaze - self.calib_center
                
                if abs(diff) > self.deadzone:
                    # Normalized speed math
                    if diff < 0: # Left
                        dist = abs(diff) - self.deadzone
                        rng = abs(self.calib_center - self.calib_left) + 1e-6
                        intensity = min(1.0, dist / rng)
                        target = (w//2) - (intensity * (w//2))
                    else: # Right
                        dist = abs(diff) - self.deadzone
                        rng = abs(self.calib_right - self.calib_center) + 1e-6
                        intensity = min(1.0, dist / rng)
                        target = (w//2) + (intensity * (w//2))
                    
                    self.x = int(self.x * 0.1 + target * 0.9)
                    self.y = int(self.y * 0.1 + (h//2) * 0.9)
        
        # --- HAND LOGIC ---
        if hand_res.gestures:
            gesture = hand_res.gestures[0][0].category_name
            lm = hand_res.hand_landmarks[0]
            
            if self.mode == "HAND" and gesture in ["Pointing_Up", "Victory"]:
                self.x = int(lm[8].x * w)
                self.y = int(lm[8].y * h)
            
            if gesture == "Open_Palm":
                angle = get_hand_rotation(lm[0], lm[9])
                if self.prev_hand_angle is not None:
                    delta = angle - self.prev_hand_angle
                    if delta > 180: delta -= 360
                    elif delta < -180: delta += 360
                    if abs(delta) > 1.0:
                        self.radius = max(20, min(300, self.radius + delta * 2.0))
                self.prev_hand_angle = angle
            else:
                self.prev_hand_angle = None

            if self.mode == "HAND" and gesture == "Closed_Fist":
                 if time.time() - self.last_fist_time > 1.0:
                    self.color_idx = (self.color_idx + 1) % len(self.colors)
                    self.last_fist_time = time.time()

        # Draw Ball
        cv2.circle(img, (int(self.x), int(self.y)), int(self.radius), self.colors[self.color_idx], -1)
        
        # Mode Indicator
        color_ind = (0, 255, 0) if self.mode == "HAND" else (0, 0, 255)
        cv2.putText(img, f"MODE: {self.mode}", (w-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_ind, 2)
        
        if self.mode == "EYE" and not self.is_calibrated:
             cv2.putText(img, "NOT CALIBRATED", (w//2-100, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. STREAMLIT UI ---
st.title("Multimodal Eye & Hand Control")

col1, col2 = st.columns([3, 1])

with col1:
    ctx = webrtc_streamer(
        key="eye-hand",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EyeHandProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("### 🎛 Controls")
    
    # Store calibration in session state so it persists
    if "calib_vals" not in st.session_state:
        st.session_state["calib_vals"] = {"L": 0.4, "C": 0.5, "R": 0.6, "Done": False}

    mode_select = st.radio("Mode", ["HAND", "EYE"])

    st.markdown("---")
    st.markdown("### 👁 Calibration")
    st.write("Look at the screen position and click:")

    # Direct interaction with the LIVE processor
    if ctx.video_processor:
        processor = ctx.video_processor
        processor.mode = mode_select
        
        # Update processor with stored calibration
        if st.session_state["calib_vals"]["Done"]:
            processor.calib_left = st.session_state["calib_vals"]["L"]
            processor.calib_center = st.session_state["calib_vals"]["C"]
            processor.calib_right = st.session_state["calib_vals"]["R"]
            processor.is_calibrated = True

        current_gaze = processor.last_raw_gaze
        st.metric("Current Gaze Ratio", f"{current_gaze:.2f}")

        if st.button("1. Look Center (Set)"):
            st.session_state["calib_vals"]["C"] = current_gaze
            st.success(f"Center set to {current_gaze:.2f}")

        if st.button("2. Look Left (Set)"):
            st.session_state["calib_vals"]["L"] = current_gaze
            st.success(f"Left set to {current_gaze:.2f}")

        if st.button("3. Look Right (Set)"):
            st.session_state["calib_vals"]["R"] = current_gaze
            st.success(f"Right set to {current_gaze:.2f}")
            
        if st.button("✅ Activate Calibration"):
            st.session_state["calib_vals"]["Done"] = True
            # Force update immediately
            processor.is_calibrated = True
            st.success("Calibration Active!")

    else:
        st.warning("Start video to enable controls.")