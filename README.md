# 👁️✋ Multimodal Neural Controller

> **Control a digital object using nothing but your Gaze and Hand Gestures.**
> No mouse. No keyboard. Just Computer Vision.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mmnc-by-nojan.streamlit.app)

**PLEASE USE PC/LAPTOP FOR BETTER EXPERIENCE**

## 🔥 The Concept
This project is a Proof-of-Concept (PoC) for **multimodal human-computer interaction (HCI)**. It leverages Google's MediaPipe framework to run two synchronous neural networks in real-time directly in your browser:

1.  **Hand Landmark Model:** Tracks 21 skeletal points on your hand to detect gestures and rotation.
2.  **Face Landmark Model:** Tracks 478 mesh points on your face, specifically isolating the iris for gaze estimation.

## 🕹️ How to Use

**Step 1: Allow Camera Access**
The app needs to see you. We do not store any video data; it is processed live in memory.

**Step 2: Choose Your Weapon (Mode)**
* **Hand Mode (The "Iron Man" Interface)**
    * **Move:** Point your Index Finger.
    * **Action:** Make a **Closed Fist** to change color.
    * **Resize:** Show an **Open Palm** and rotate your wrist like a volume knob.
* **Eye Mode (The "Telekinesis" Interface)**
    * **Move:** Look **Left** or **Right**.
    * **Action:** **Blink** intentionally to change color.
    * **Resize:** Use the Hand (Open Palm) gesture simultaneously.

**Step 3: CALIBRATE (Crucial)**
Eye tracking varies by person and lighting.
1.  Select **Eye Mode**.
2.  Look at the screen center and click **"Look Center (Set)"**.
3.  Look at the left edge and click **"Look Left (Set)"**.
4.  Look at the right edge and click **"Look Right (Set)"**.
5.  Click **"Activate Calibration"**.

---

## ⚠️ The Reality Check (Limitations)

Let's be real. Eye tracking with a standard webcam is **hard**. This software pushes the limits of what non-infrared cameras can do.

### 1. The "Jitter" Problem
Your eyes naturally make micro-movements (saccades) constantly. Even with our exponential smoothing algorithms, the cursor *will* shake. It's biological, not just algorithmic.

### 2. Lighting is God
* **Dark Room?** The model will lose your iris.
* **Backlit?** The model will lose your face.
* **Glasses?** Reflections will confuse the mesh.
* **Fix:** Shine a light directly on your face. If the yellow dots on your eyes disappear, the system is blind.

### 3. Calibration Drift
You must calibrate every time you sit down. If you move your head significantly after calibration, the mapping breaks. The system assumes your head is relatively stationary and only your eyes are moving.

### 4. "Ghost" Blinks
If you look down, your eyelids naturally lower. The system *will* think you blinked. The math uses Aspect Ratio (EAR); it cannot distinguish between "Looking Down" and "Blinking."

---

## 🧠 Under the Hood (The Math)

### Hand Rotation Logic
We calculate the arctangent of the vector between the **Wrist** (Landmark 0) and the **Middle Finger MCP** (Landmark 9).
$$\theta = \arctan2(y_9 - y_0, x_9 - x_0)$$
We track the delta ($\Delta\theta$) between frames to drive the radius change.

### Gaze Estimation Logic
We use **Inverse Linear Interpolation** on the horizontal axis of the eye.
$$Ratio = \frac{x_{iris} - x_{outer}}{x_{inner} - x_{outer}}$$
* Ratio $\approx 0.35$ → Looking Left
* Ratio $\approx 0.65$ → Looking Right
* We apply a **Deadzone Filter** ($|\Delta| < 0.05$) to ignore micro-jitters.

---

**Built with Streamlit & MediaPipe**