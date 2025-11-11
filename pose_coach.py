import cv2
import numpy as np
import time
import csv
from datetime import datetime
from collections import deque
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk


class PoseCoachGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Coach - AI Form Trainer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')

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

        # Timing
        self.session_start = time.time()
        self.rep_start_time = None
        self.rep_times = []
        self.rest_start = None
        self.rest_time = 0

        # Form feedback
        self.feedback_msg = ""
        self.angle_history = deque(maxlen=5)

        # Feedback timing - keep messages visible longer
        self.last_feedback_time = time.time()
        self.last_feedback_msg = ""
        self.feedback_cooldown = 1.5  # Show each message for at least 1.5 seconds

        # Recording and camera
        self.recording = False
        self.video_writer = None
        self.camera_active = False
        self.cap = None

        # Audio
        self.audio_enabled = tk.BooleanVar(value=False)

        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()

        # Angle threshold
        self.depth_threshold = tk.IntVar(value=90)

        # Feedback display duration
        self.feedback_duration = tk.DoubleVar(value=1.5)

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        """Create the GUI layout"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Video feed
        left_panel = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Video label
        self.video_label = tk.Label(left_panel, bg='#000000', text="Camera Feed",
                                    fg='white', font=('Arial', 20))
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right panel - Controls and stats
        right_panel = tk.Frame(main_frame, bg='#2d2d2d', width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)

        # Title
        title_label = tk.Label(right_panel, text="POSE COACH",
                               font=('Arial', 20, 'bold'), bg='#2d2d2d', fg='#00ff88')
        title_label.pack(pady=(10, 20))

        # Control buttons frame
        controls_frame = tk.LabelFrame(right_panel, text="Controls",
                                       font=('Arial', 12, 'bold'),
                                       bg='#2d2d2d', fg='white', relief=tk.RIDGE, bd=2)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        # Start/Stop button
        self.start_button = tk.Button(controls_frame, text="▶ START CAMERA",
                                      command=self.toggle_camera,
                                      bg='#00ff88', fg='black',
                                      font=('Arial', 12, 'bold'),
                                      height=2, cursor='hand2')
        self.start_button.pack(fill=tk.X, padx=10, pady=5)

        # Exercise switch
        exercise_frame = tk.Frame(controls_frame, bg='#2d2d2d')
        exercise_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(exercise_frame, text="Exercise:", bg='#2d2d2d',
                 fg='white', font=('Arial', 10)).pack(side=tk.LEFT)

        self.exercise_var = tk.StringVar(value="squat")
        exercise_menu = ttk.Combobox(exercise_frame, textvariable=self.exercise_var,
                                     values=["squat", "pushup", "bicep_curl"], state='readonly',
                                     font=('Arial', 10), width=15)
        exercise_menu.pack(side=tk.RIGHT)
        exercise_menu.bind('<<ComboboxSelected>>', self.change_exercise)

        # Action buttons
        btn_frame = tk.Frame(controls_frame, bg='#2d2d2d')
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.record_button = tk.Button(btn_frame, text="RECORD",
                                       command=self.toggle_recording,
                                       bg='#ff4444', fg='white',
                                       font=('Arial', 10, 'bold'),
                                       cursor='hand2', state=tk.DISABLED)
        self.record_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        new_set_button = tk.Button(btn_frame, text="NEW SET",
                                   command=self.new_set,
                                   bg='#4444ff', fg='white',
                                   font=('Arial', 10, 'bold'),
                                   cursor='hand2')
        new_set_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        # Stats frame
        stats_frame = tk.LabelFrame(right_panel, text="Live Statistics",
                                    font=('Arial', 12, 'bold'),
                                    bg='#2d2d2d', fg='white', relief=tk.RIDGE, bd=2)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)

        # Stats display
        stats_grid = tk.Frame(stats_frame, bg='#2d2d2d')
        stats_grid.pack(fill=tk.X, padx=10, pady=10)

        # Reps
        tk.Label(stats_grid, text="Reps:", bg='#2d2d2d', fg='#aaaaaa',
                 font=('Arial', 10)).grid(row=0, column=0, sticky='w', pady=2)
        self.reps_label = tk.Label(stats_grid, text="0", bg='#2d2d2d',
                                   fg='#00ff88', font=('Arial', 24, 'bold'))
        self.reps_label.grid(row=0, column=1, sticky='e', pady=2)

        # Sets
        tk.Label(stats_grid, text="Set:", bg='#2d2d2d', fg='#aaaaaa',
                 font=('Arial', 10)).grid(row=1, column=0, sticky='w', pady=2)
        self.sets_label = tk.Label(stats_grid, text="1", bg='#2d2d2d',
                                   fg='#ffffff', font=('Arial', 18, 'bold'))
        self.sets_label.grid(row=1, column=1, sticky='e', pady=2)

        # Stage
        tk.Label(stats_grid, text="Stage:", bg='#2d2d2d', fg='#aaaaaa',
                 font=('Arial', 10)).grid(row=2, column=0, sticky='w', pady=2)
        self.stage_label = tk.Label(stats_grid, text="N/A", bg='#2d2d2d',
                                    fg='#ffaa00', font=('Arial', 14, 'bold'))
        self.stage_label.grid(row=2, column=1, sticky='e', pady=2)

        # Active arm (for bicep curls)
        tk.Label(stats_grid, text="Arm:", bg='#2d2d2d', fg='#aaaaaa',
                 font=('Arial', 10)).grid(row=3, column=0, sticky='w', pady=2)
        self.arm_label = tk.Label(stats_grid, text="--", bg='#2d2d2d',
                                  fg='#ffffff', font=('Arial', 12))
        self.arm_label.grid(row=3, column=1, sticky='e', pady=2)

        # Current Angle
        tk.Label(stats_grid, text="Angle:", bg='#2d2d2d', fg='#aaaaaa',
                 font=('Arial', 10)).grid(row=4, column=0, sticky='w', pady=2)
        self.angle_label = tk.Label(stats_grid, text="--°", bg='#2d2d2d',
                                    fg='#ffffff', font=('Arial', 14))
        self.angle_label.grid(row=4, column=1, sticky='e', pady=2)

        # Best Angle
        tk.Label(stats_grid, text="Best:", bg='#2d2d2d', fg='#aaaaaa',
                 font=('Arial', 10)).grid(row=5, column=0, sticky='w', pady=2)
        self.best_label = tk.Label(stats_grid, text="--°", bg='#2d2d2d',
                                   fg='#00ffff', font=('Arial', 14))
        self.best_label.grid(row=5, column=1, sticky='e', pady=2)

        # FPS
        tk.Label(stats_grid, text="FPS:", bg='#2d2d2d', fg='#aaaaaa',
                 font=('Arial', 10)).grid(row=6, column=0, sticky='w', pady=2)
        self.fps_label = tk.Label(stats_grid, text="--", bg='#2d2d2d',
                                  fg='#ffffff', font=('Arial', 14))
        self.fps_label.grid(row=6, column=1, sticky='e', pady=2)

        stats_grid.columnconfigure(1, weight=1)

        # Feedback frame
        feedback_frame = tk.LabelFrame(right_panel, text="Form Feedback",
                                       font=('Arial', 12, 'bold'),
                                       bg='#2d2d2d', fg='white', relief=tk.RIDGE, bd=2)
        feedback_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.feedback_label = tk.Label(feedback_frame, text="Ready to start!",
                                       bg='#2d2d2d', fg='#ffaa00',
                                       font=('Arial', 16, 'bold'),
                                       wraplength=300, justify='center',
                                       height=4)
        self.feedback_label.pack(padx=10, pady=20, fill=tk.BOTH, expand=True)

        # Settings frame
        settings_frame = tk.LabelFrame(right_panel, text="Settings",
                                       font=('Arial', 12, 'bold'),
                                       bg='#2d2d2d', fg='white', relief=tk.RIDGE, bd=2)
        settings_frame.pack(fill=tk.X, padx=10, pady=10)

        # Depth threshold
        depth_frame = tk.Frame(settings_frame, bg='#2d2d2d')
        depth_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(depth_frame, text="Depth Threshold:", bg='#2d2d2d',
                 fg='white', font=('Arial', 9)).pack(anchor='w')

        depth_slider = tk.Scale(depth_frame, from_=70, to=110,
                                variable=self.depth_threshold,
                                orient=tk.HORIZONTAL, bg='#2d2d2d',
                                fg='white', highlightthickness=0,
                                troughcolor='#444444', activebackground='#00ff88')
        depth_slider.pack(fill=tk.X)

        # Feedback duration
        feedback_frame = tk.Frame(settings_frame, bg='#2d2d2d')
        feedback_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(feedback_frame, text="Feedback Speed (seconds):", bg='#2d2d2d',
                 fg='white', font=('Arial', 9)).pack(anchor='w')

        feedback_slider = tk.Scale(feedback_frame, from_=0.5, to=3.0,
                                   resolution=0.5,
                                   variable=self.feedback_duration,
                                   orient=tk.HORIZONTAL, bg='#2d2d2d',
                                   fg='white', highlightthickness=0,
                                   troughcolor='#444444', activebackground='#00ff88',
                                   command=self.update_feedback_cooldown)
        feedback_slider.pack(fill=tk.X)

        # Audio toggle
        audio_check = tk.Checkbutton(settings_frame, text="Audio Cues",
                                     variable=self.audio_enabled,
                                     bg='#2d2d2d', fg='white',
                                     selectcolor='#1e1e1e',
                                     font=('Arial', 10),
                                     activebackground='#2d2d2d')
        audio_check.pack(anchor='w', padx=10, pady=5)

        # Save session button
        save_button = tk.Button(right_panel, text="SAVE SESSION",
                                command=self.save_session,
                                bg='#ff8800', fg='white',
                                font=('Arial', 11, 'bold'),
                                cursor='hand2', height=2)
        save_button.pack(fill=tk.X, padx=10, pady=10, side=tk.BOTTOM)

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

    def get_smoothed_angle(self, angle):
        """Smooth angle using moving average"""
        self.angle_history.append(angle)
        return np.mean(self.angle_history)

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
                cv2.line(image, start_point, end_point, (0, 255, 0), 3)

        for i, kp in enumerate(keypoints):
            if kp[2] > 0.5:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                cv2.circle(image, (x, y), 7, (0, 255, 255), 2)

    def get_keypoint(self, keypoints, name):
        """Get keypoint coordinates by name"""
        idx = self.KEYPOINTS[name]
        return keypoints[idx][:2]

    def update_feedback_cooldown(self, value):
        """Update the feedback cooldown when slider changes"""
        self.feedback_cooldown = float(value)

    def update_feedback(self, new_feedback):
        """Update feedback with cooldown to prevent rapid changes"""
        current_time = time.time()

        # If feedback is different and cooldown has passed, update it
        if new_feedback != self.last_feedback_msg:
            if current_time - self.last_feedback_time >= self.feedback_cooldown:
                self.feedback_msg = new_feedback
                self.last_feedback_msg = new_feedback
                self.last_feedback_time = current_time
            # Otherwise keep showing the last message
            else:
                self.feedback_msg = self.last_feedback_msg
        else:
            # Same message, just keep showing it
            self.feedback_msg = new_feedback

    def detect_bicep_curl(self, keypoints):
        """Detect bicep curl reps and provide form feedback"""
        # Detect both arms and use the one with better visibility or more flexion
        left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        left_elbow = self.get_keypoint(keypoints, 'left_elbow')
        left_wrist = self.get_keypoint(keypoints, 'left_wrist')

        right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
        right_elbow = self.get_keypoint(keypoints, 'right_elbow')
        right_wrist = self.get_keypoint(keypoints, 'right_wrist')

        # Check visibility/confidence for both arms
        left_conf = (keypoints[self.KEYPOINTS['left_shoulder']][2] +
                     keypoints[self.KEYPOINTS['left_elbow']][2] +
                     keypoints[self.KEYPOINTS['left_wrist']][2]) / 3

        right_conf = (keypoints[self.KEYPOINTS['right_shoulder']][2] +
                      keypoints[self.KEYPOINTS['right_elbow']][2] +
                      keypoints[self.KEYPOINTS['right_wrist']][2]) / 3

        # Calculate angles for both arms
        left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Choose arm based on which one is MORE FLEXED (lower angle = more curled)
        # This ensures we track the arm that's actively curling
        use_right = False

        # If both arms visible, use the one more flexed
        if left_conf > 0.5 and right_conf > 0.5:
            # Use the arm with the smaller angle (more flexed)
            use_right = right_angle < left_angle
        elif right_conf > 0.6:  # Only right arm clearly visible
            use_right = True
        elif left_conf > 0.6:  # Only left arm clearly visible
            use_right = False
        else:  # Both weak, use better confidence
            use_right = right_conf > left_conf

        if use_right:
            shoulder = right_shoulder
            elbow = right_elbow
            wrist = right_wrist
            angle = right_angle
            self.curl_side = "right"
        else:
            shoulder = left_shoulder
            elbow = left_elbow
            wrist = left_wrist
            angle = left_angle
            self.curl_side = "left"

        self.current_angle = self.get_smoothed_angle(angle)

        # Check elbow stability (should stay relatively close to body)
        elbow_drift = abs(elbow[0] - shoulder[0]) * 100

        # Rep detection logic - curl has opposite angle logic
        # Extended arm = high angle (~160-180°)
        # Curled arm = low angle (~30-50°)
        if self.current_angle > 150:  # Arm extended
            self.stage = "down"
            if self.rest_start:
                self.rest_time += time.time() - self.rest_start
                self.rest_start = None

        if self.current_angle < 50 and self.stage == 'down':  # Arm fully curled
            self.stage = "up"
            self.rep_count += 1

            if self.rep_start_time:
                rep_duration = time.time() - self.rep_start_time
                self.rep_times.append(rep_duration)
            self.rep_start_time = time.time()

            # For bicep curls, best angle is the SMALLEST angle (full contraction)
            if self.best_angle is None or self.current_angle < self.best_angle:
                self.best_angle = self.current_angle

            if self.audio_enabled.get():
                print('\a')

            self.rest_start = time.time()

        # Form feedback
        feedback = ""
        if self.stage == "up":
            if self.current_angle > 70:
                feedback = "CURL MORE!"
            elif self.current_angle < 30:
                feedback = "GREAT CONTRACTION!"
            else:
                feedback = "GOOD CURL!"
        elif self.stage == "down":
            if self.current_angle < 140:
                feedback = "EXTEND FULLY!"
            else:
                feedback = "GOOD EXTENSION!"

        # Check for elbow swing (bad form)
        if elbow_drift > 25:
            feedback += "\nKEEP ELBOW STABLE!"

        self.update_feedback(feedback)

    def detect_squat(self, keypoints):
        """Detect squat reps and provide form feedback"""
        hip = self.get_keypoint(keypoints, 'left_hip')
        knee = self.get_keypoint(keypoints, 'left_knee')
        ankle = self.get_keypoint(keypoints, 'left_ankle')
        shoulder = self.get_keypoint(keypoints, 'left_shoulder')

        angle = self.calculate_angle(hip, knee, ankle)
        self.current_angle = self.get_smoothed_angle(angle)

        back_angle = abs(hip[0] - shoulder[0]) * 100

        threshold = self.depth_threshold.get()

        if self.current_angle > 160:
            self.stage = "up"
            if self.rest_start:
                self.rest_time += time.time() - self.rest_start
                self.rest_start = None

        if self.current_angle < threshold and self.stage == 'up':
            self.stage = "down"
            self.rep_count += 1

            if self.rep_start_time:
                rep_duration = time.time() - self.rep_start_time
                self.rep_times.append(rep_duration)
            self.rep_start_time = time.time()

            if self.best_angle is None or self.current_angle < self.best_angle:
                self.best_angle = self.current_angle

            if self.audio_enabled.get():
                print('\a')

            self.rest_start = time.time()

        self.feedback_msg = ""
        if self.stage == "down":
            if self.current_angle > threshold + 10:
                feedback = "GO DEEPER!"
            elif self.current_angle < 70:
                feedback = "TOO LOW!"
            else:
                feedback = "GOOD DEPTH!"
        else:
            feedback = ""

        if back_angle > 15:
            feedback += "\nKEEP BACK STRAIGHT!"

        self.update_feedback(feedback)

    def detect_pushup(self, keypoints):
        """Detect pushup reps and provide form feedback"""
        shoulder = self.get_keypoint(keypoints, 'left_shoulder')
        elbow = self.get_keypoint(keypoints, 'left_elbow')
        wrist = self.get_keypoint(keypoints, 'left_wrist')
        hip = self.get_keypoint(keypoints, 'left_hip')

        angle = self.calculate_angle(shoulder, elbow, wrist)
        self.current_angle = self.get_smoothed_angle(angle)

        body_angle = abs(shoulder[1] - hip[1]) * 100

        threshold = self.depth_threshold.get()

        if self.current_angle > 160:
            self.stage = "up"
            if self.rest_start:
                self.rest_time += time.time() - self.rest_start
                self.rest_start = None

        if self.current_angle < threshold and self.stage == 'up':
            self.stage = "down"
            self.rep_count += 1

            if self.rep_start_time:
                rep_duration = time.time() - self.rep_start_time
                self.rep_times.append(rep_duration)
            self.rep_start_time = time.time()

            if self.best_angle is None or self.current_angle < self.best_angle:
                self.best_angle = self.current_angle

            if self.audio_enabled.get():
                print('\a')

            self.rest_start = time.time()

        self.feedback_msg = ""
        if self.stage == "down":
            if self.current_angle > threshold + 10:
                feedback = "GO LOWER!"
            elif self.current_angle < 70:
                feedback = "TOO LOW!"
            else:
                feedback = "GOOD FORM!"
        else:
            feedback = ""

        if body_angle > 20:
            feedback += "\nKEEP BODY STRAIGHT!"

        self.update_feedback(feedback)

    def update_stats_display(self):
        """Update the statistics labels"""
        self.reps_label.config(text=str(self.rep_count))
        self.sets_label.config(text=str(self.set_count))
        self.stage_label.config(text=self.stage.upper() if self.stage else "N/A")

        # Show which arm is being tracked for bicep curls
        if self.exercise_type == "bicep_curl":
            self.arm_label.config(text=self.curl_side.upper())
        else:
            self.arm_label.config(text="--")

        if self.current_angle:
            self.angle_label.config(text=f"{self.current_angle:.1f}°")

        if self.best_angle:
            self.best_label.config(text=f"{self.best_angle:.1f}°")

        if self.fps_history:
            self.fps_label.config(text=f"{np.mean(self.fps_history):.1f}")

        self.feedback_label.config(text=self.feedback_msg if self.feedback_msg else "Looking good!")

    def process_frame(self):
        """Process camera frame"""
        if not self.camera_active or self.cap is None:
            return

        success, image = self.cap.read()
        if not success:
            return

        image = cv2.flip(image, 1)

        current_time = time.time()
        fps = 1 / (current_time - self.last_time)
        self.fps_history.append(fps)
        self.last_time = current_time

        results = self.model(image, verbose=False)

        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            keypoints = results[0].keypoints.data[0].cpu().numpy()

            self.draw_skeleton(image, keypoints)

            # For bicep curls, show both arm angles on screen for debugging
            if self.exercise_type == "bicep_curl":
                left_shoulder = self.get_keypoint(keypoints, 'left_shoulder')
                left_elbow = self.get_keypoint(keypoints, 'left_elbow')
                left_wrist = self.get_keypoint(keypoints, 'left_wrist')
                right_shoulder = self.get_keypoint(keypoints, 'right_shoulder')
                right_elbow = self.get_keypoint(keypoints, 'right_elbow')
                right_wrist = self.get_keypoint(keypoints, 'right_wrist')

                left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Draw angle info on video
                h, w = image.shape[:2]
                cv2.putText(image, f"Left: {left_angle:.1f}°", (10, h - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(image, f"Right: {right_angle:.1f}°", (10, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if self.exercise_type == "squat":
                required = ['left_hip', 'left_knee', 'left_ankle']
                all_visible = all(keypoints[self.KEYPOINTS[kp]][2] > 0.5 for kp in required)
            elif self.exercise_type == "pushup":
                required = ['left_shoulder', 'left_elbow', 'left_wrist']
                all_visible = all(keypoints[self.KEYPOINTS[kp]][2] > 0.5 for kp in required)
            else:  # bicep_curl - check both arms
                left_visible = all(keypoints[self.KEYPOINTS[kp]][2] > 0.5
                                   for kp in ['left_shoulder', 'left_elbow', 'left_wrist'])
                right_visible = all(keypoints[self.KEYPOINTS[kp]][2] > 0.5
                                    for kp in ['right_shoulder', 'right_elbow', 'right_wrist'])
                all_visible = left_visible or right_visible

            if all_visible:
                if self.exercise_type == "squat":
                    self.detect_squat(keypoints)
                elif self.exercise_type == "pushup":
                    self.detect_pushup(keypoints)
                else:  # bicep_curl
                    self.detect_bicep_curl(keypoints)
            else:
                self.feedback_msg = "⚠ ADJUST POSITION"
        else:
            self.feedback_msg = "⚠ NO PERSON DETECTED"

        self.update_stats_display()

        # Draw feedback message on video feed for better visibility
        if self.feedback_msg:
            h, w = image.shape[:2]
            # Draw semi-transparent background
            overlay = image.copy()
            cv2.rectangle(overlay, (10, h - 150), (w - 10, h - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            # Split message by newline and draw each line
            lines = self.feedback_msg.split('\n')
            y_offset = h - 110
            for line in lines:
                # Choose color based on message
                if 'GOOD' in line or 'GREAT' in line:
                    color = (0, 255, 0)  # Green
                elif 'TOO' in line or 'WARNING' in line:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 255, 255)  # Yellow

                cv2.putText(image, line, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                y_offset += 50

        if self.recording and self.video_writer:
            self.video_writer.write(image)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil = image_pil.resize((960, 720), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image_pil)

        self.video_label.config(image=photo)
        self.video_label.image = photo

        self.root.after(10, self.process_frame)

    def toggle_camera(self):
        """Start or stop camera"""
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera_active = True
            self.start_button.config(text="⏹ STOP CAMERA", bg='#ff4444')
            self.record_button.config(state=tk.NORMAL)
            self.process_frame()
        else:
            self.camera_active = False
            if self.cap:
                self.cap.release()
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            self.start_button.config(text="▶ START CAMERA", bg='#00ff88')
            self.record_button.config(text="RECORD", bg='#ff4444', state=tk.DISABLED)
            self.video_label.config(image='', text="Camera Stopped", fg='white')

    def toggle_recording(self):
        """Start or stop recording"""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pose_coach_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (1280, 720))
            self.recording = True
            self.record_button.config(text="STOP REC", bg='#00ff00')
            messagebox.showinfo("Recording", f"Recording started: {filename}")
        else:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.record_button.config(text="RECORD", bg='#ff4444')
            messagebox.showinfo("Recording", "Recording stopped and saved!")

    def change_exercise(self, event=None):
        """Change exercise type"""
        self.exercise_type = self.exercise_var.get()
        self.rep_count = 0
        self.stage = None
        self.best_angle = None
        self.angle_history.clear()
        self.update_stats_display()

    def new_set(self):
        """Start a new set"""
        self.set_count += 1
        self.rep_count = 0
        self.stage = None
        self.update_stats_display()
        messagebox.showinfo("New Set", f"Starting Set {self.set_count}")

    def save_session(self):
        """Save session data to CSV"""
        if self.rep_count == 0:
            messagebox.showwarning("No Data", "No reps recorded yet!")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pose_coach_session_{timestamp}.csv"

        avg_rep_time = np.mean(self.rep_times) if self.rep_times else 0
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        session_duration = time.time() - self.session_start

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Session Summary'])
            writer.writerow(['Exercise Type', self.exercise_type])
            writer.writerow(['Total Reps', self.rep_count])
            writer.writerow(['Sets Completed', self.set_count])
            writer.writerow(['Best Angle', f"{self.best_angle:.1f}°" if self.best_angle else "N/A"])
            writer.writerow(['Average Rep Time', f"{avg_rep_time:.2f}s"])
            writer.writerow(['Total Rest Time', f"{self.rest_time:.1f}s"])
            writer.writerow(['Session Duration', f"{session_duration:.1f}s"])
            writer.writerow(['Average FPS', f"{avg_fps:.1f}"])
            writer.writerow([])
            writer.writerow(['Individual Rep Times (seconds)'])
            for i, rep_time in enumerate(self.rep_times, 1):
                writer.writerow([f"Rep {i}", f"{rep_time:.2f}"])

        messagebox.showinfo("Session Saved", f"Session data saved to:\n{filename}")

    def on_closing(self):
        """Handle window closing"""
        if self.camera_active:
            self.camera_active = False
            if self.cap:
                self.cap.release()
            if self.video_writer:
                self.video_writer.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PoseCoachGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()