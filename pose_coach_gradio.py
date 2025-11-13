import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
import gradio as gr
from PIL import Image

class PoseCoachGradio:
    def __init__(self):
        # Load YOLOv8 Pose model
        print("Loading YOLOv8-Pose model...")
        self.model = YOLO('yolov8n-pose.pt')
        print("Model loaded successfully!")

        # YOLO pose keypoint indices
        self.KEYPOINTS = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }

        # Exercise state
        self.exercise_type = "squat"
        self.rep_count = 0
        self.set_count = 1
        self.stage = None
        self.best_angle = None
        self.current_angle = None
        self.curl_side = "left"

        # Camera state
        self.camera_active = False
        self.cap = None

        # Form feedback
        self.feedback_msg = ""
        self.angle_history = deque(maxlen=5)

    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def draw_skeleton(self, image, keypoints):
        """Draw skeleton connections on image"""
        skeleton = [
            (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        for start_idx, end_idx in skeleton:
            if keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5:
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        for i, kp in enumerate(keypoints):
            if kp[2] > 0.5:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                cv2.circle(image, (x, y), 7, (0, 255, 255), 1)

    def start_camera(self):
        """Start the camera"""
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera_active = True
            return "Camera started"
        return "Camera is already active"

    def stop_camera(self):
        """Stop the camera"""
        if self.camera_active:
            self.camera_active = False
            if self.cap:
                self.cap.release()
            return "Camera stopped"
        return "Camera is not active"

    def process_frame(self, frame):
        """Process a single frame"""
        if not self.camera_active or self.cap is None:
            return frame, "Camera not active", "No stats yet"

        image = cv2.flip(frame, 1)
        results = self.model(image, verbose=False)

        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            self.draw_skeleton(image, keypoints)

            # Detect exercise based on keypoints
            if self.exercise_type == "squat":
                required = ['left_hip', 'left_knee', 'left_ankle']
                all_visible = all(keypoints[self.KEYPOINTS[kp]][2] > 0.5 for kp in required)
                if all_visible:
                    self.detect_squat(keypoints)
            elif self.exercise_type == "pushup":
                required = ['left_shoulder', 'left_elbow', 'left_wrist']
                all_visible = all(keypoints[self.KEYPOINTS[kp]][2] > 0.5 for kp in required)
                if all_visible:
                    self.detect_pushup(keypoints)
            else:  # bicep_curl
                left_visible = all(keypoints[self.KEYPOINTS[kp]][2] > 0.5
                                   for kp in ['left_shoulder', 'left_elbow', 'left_wrist'])
                right_visible = all(keypoints[self.KEYPOINTS[kp]][2] > 0.5
                                    for kp in ['right_shoulder', 'right_elbow', 'right_wrist'])
                all_visible = left_visible or right_visible
                if all_visible:
                    self.detect_bicep_curl(keypoints)

        # Convert frame to RGB for Gradio
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        stats = (
            f"Reps: {self.rep_count}\n"
            f"Set: {self.set_count}\n"
            f"Stage: {self.stage}\n"
            f"Angle: {self.current_angle:.1f}°\n" if self.current_angle is not None else "Angle: N/A\n"
            f"Best Angle: {self.best_angle:.1f}°\n" if self.best_angle is not None else "Best Angle: N/A\n"
        )
        feedback = self.feedback_msg if self.feedback_msg else "No feedback yet"

        return frame_pil, stats, feedback

    def get_keypoint(self, keypoints, name):
        """Get keypoint coordinates by name"""
        idx = self.KEYPOINTS[name]
        return keypoints[idx][:2]

    def detect_bicep_curl(self, keypoints):
        """Detect bicep curl reps and provide form feedback"""
        left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        left_elbow = self.get_keypoint(keypoints, 'left_elbow')
        left_wrist = self.get_keypoint(keypoints, 'left_wrist')
        right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
        right_elbow = self.get_keypoint(keypoints, 'right_elbow')
        right_wrist = self.get_keypoint(keypoints, 'right_wrist')

        left_conf = (keypoints[self.KEYPOINTS['left_shoulder']][2] +
                     keypoints[self.KEYPOINTS['left_elbow']][2] +
                     keypoints[self.KEYPOINTS['left_wrist']][2]) / 3
        right_conf = (keypoints[self.KEYPOINTS['right_shoulder']][2] +
                      keypoints[self.KEYPOINTS['right_elbow']][2] +
                      keypoints[self.KEYPOINTS['right_wrist']][2]) / 3

        left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

        use_right = False
        if left_conf > 0.5 and right_conf > 0.5:
            use_right = right_angle < left_angle
        elif right_conf > 0.6:
            use_right = True
        elif left_conf > 0.6:
            use_right = False
        else:
            use_right = right_conf > left_conf

        if use_right:
            shoulder = right_shoulder
            elbow = right_elbow
            wrist = right_wrist
            angle = right_angle
            self.curl_side = "left"
        else:
            shoulder = left_shoulder
            elbow = left_elbow
            wrist = left_wrist
            angle = left_angle
            self.curl_side = "right"

        self.current_angle = self.get_smoothed_angle(angle)
        elbow_drift = abs(elbow[0] - shoulder[0]) * 100

        if self.current_angle > 160:
            if self.stage != "extended":
                self.stage = "extended"
        elif self.current_angle < 60:
            if self.stage == "extended":
                self.stage = "curled"
                self.rep_count += 1
                if self.best_angle is None or self.current_angle < self.best_angle:
                    self.best_angle = self.current_angle
        elif 60 <= self.current_angle <= 160:
            if self.stage not in ["extended", "curled"]:
                self.stage = "transitioning"

        feedback = ""
        if self.stage == "curled":
            if self.current_angle > 80:
                feedback = "CURL MORE!"
            elif self.current_angle < 40:
                feedback = "GREAT CONTRACTION!"
            else:
                feedback = "GOOD CURL!"
        elif self.stage == "extended":
            feedback = "READY - CURL NOW!"
        elif self.stage == "transitioning":
            if self.current_angle < 100:
                feedback = "KEEP CURLING!"
            else:
                feedback = "EXTEND FULLY!"

        if elbow_drift > 25:
            feedback += "\nKEEP ELBOW STABLE!"

        self.feedback_msg = feedback

    def detect_squat(self, keypoints):
        """Detect squat reps and provide form feedback"""
        hip = self.get_keypoint(keypoints, 'left_hip')
        knee = self.get_keypoint(keypoints, 'left_knee')
        ankle = self.get_keypoint(keypoints, 'left_ankle')
        shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        angle = self.calculate_angle(hip, knee, ankle)
        self.current_angle = self.get_smoothed_angle(angle)
        back_angle = abs(hip[0] - shoulder[0]) * 100

        if self.current_angle > 160:
            self.stage = "up"
        if self.current_angle < 90 and self.stage == 'up':
            self.stage = "down"
            self.rep_count += 1
            if self.best_angle is None or self.current_angle < self.best_angle:
                self.best_angle = self.current_angle

        feedback = ""
        if self.stage == "down":
            if self.current_angle > 100:
                feedback = "GO DEEPER!"
            elif self.current_angle < 70:
                feedback = "TOO LOW!"
            else:
                feedback = "GOOD DEPTH!"
        if back_angle > 15:
            feedback += "\nKEEP BACK STRAIGHT!"

        self.feedback_msg = feedback

    def detect_pushup(self, keypoints):
        """Detect pushup reps and provide form feedback"""
        shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        elbow = self.get_keypoint(keypoints, 'left_elbow')
        wrist = self.get_keypoint(keypoints, 'left_wrist')
        hip = self.get_keypoint(keypoints, 'left_hip')
        angle = self.calculate_angle(shoulder, elbow, wrist)
        self.current_angle = self.get_smoothed_angle(angle)
        body_angle = abs(shoulder[1] - hip[1]) * 100

        if self.current_angle > 160:
            self.stage = "up"
        if self.current_angle < 90 and self.stage == 'up':
            self.stage = "down"
            self.rep_count += 1
            if self.best_angle is None or self.current_angle < self.best_angle:
                self.best_angle = self.current_angle

        feedback = ""
        if self.stage == "down":
            if self.current_angle > 100:
                feedback = "GO LOWER!"
            elif self.current_angle < 70:
                feedback = "TOO LOW!"
            else:
                feedback = "GOOD FORM!"
        if body_angle > 20:
            feedback += "\nKEEP BODY STRAIGHT!"

        self.feedback_msg = feedback

    def get_smoothed_angle(self, angle):
        """Smooth angle using moving average"""
        self.angle_history.append(angle)
        return np.mean(self.angle_history)

# Create and launch the Gradio interface
def create_gradio_app():
    pose_coach = PoseCoachGradio()

    with gr.Blocks(title="Pose Coach - Gradio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Pose Coach - Gradio")
        with gr.Row():
            with gr.Column():
                camera_feed = gr.Image(label="Camera Feed", interactive=False)
                start_camera_btn = gr.Button("Start Camera")
                stop_camera_btn = gr.Button("Stop Camera", visible=False)
            with gr.Column():
                exercise_dropdown = gr.Dropdown(
                    ["squat", "pushup", "bicep_curl"],
                    value="squat",
                    label="Exercise Type"
                )
                change_exercise_btn = gr.Button("Change Exercise")
                new_set_btn = gr.Button("New Set")
                stats_display = gr.Textbox(label="Live Statistics", lines=5, interactive=False)
                feedback_display = gr.Textbox(label="Form Feedback", lines=3, interactive=False)

        def start_camera():
            result = pose_coach.start_camera()
            start_camera_btn.visible = False
            stop_camera_btn.visible = True
            return result

        def stop_camera():
            result = pose_coach.stop_camera()
            start_camera_btn.visible = True
            stop_camera_btn.visible = False
            return result

        def change_exercise(exercise):
            pose_coach.exercise_type = exercise
            pose_coach.rep_count = 0
            pose_coach.stage = None
            pose_coach.best_angle = None
            pose_coach.angle_history.clear()
            return f"Exercise changed to {exercise}"

        def new_set():
            pose_coach.set_count += 1
            pose_coach.rep_count = 0
            pose_coach.stage = None
            return f"Starting Set {pose_coach.set_count}"

        def update_ui():
            while True:
                if pose_coach.camera_active and pose_coach.cap is not None:
                    success, frame = pose_coach.cap.read()
                    if success:
                        processed_frame, stats, feedback = pose_coach.process_frame(frame)
                        yield processed_frame, stats, feedback
                    else:
                        yield Image.new('RGB', (640, 480), color='black'), "Camera error", "No feedback"
                else:
                    yield Image.new('RGB', (640, 480), color='black'), "Camera not active", "No feedback"
                time.sleep(0.1)

        start_camera_btn.click(start_camera, outputs=feedback_display)
        stop_camera_btn.click(stop_camera, outputs=feedback_display)
        change_exercise_btn.click(change_exercise, inputs=exercise_dropdown, outputs=feedback_display)
        new_set_btn.click(new_set, outputs=feedback_display)

        demo.load(update_ui, outputs=[camera_feed, stats_display, feedback_display])

    return demo

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch()
