import cv2
import numpy as np
from ultralytics import YOLO
import time
import collections
import math
import torch

# ==========================================
# GOD-EYE TACTICAL OS V8.2 (FIXED DEVICE)
# ==========================================
MODEL_PATH = "yolov8n-pose.pt"


class GodEyeV8:
    def __init__(self):
        print("⚡ ANALYZING HARDWARE...")

        # --- PERBAIKAN DEVICE STRING ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")  # String yang benar untuk GPU
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU DETECTED: {device_name}. ACCELERATING...")
        else:
            self.device = torch.device("cpu")
            print("⚠️ GPU NOT FOUND. FALLING BACK TO CPU.")

        # Load model ke device
        try:
            self.model = YOLO(MODEL_PATH).to(self.device)
            print("✅ Model loaded to device successfully.")
        except Exception as e:
            print(f"❌ Error loading model to GPU: {e}. Switching to CPU...")
            self.model = YOLO(MODEL_PATH).to("cpu")
            self.device = "cpu"

        self.trail_buffer = collections.deque(maxlen=10)
        self.prev_kpts = {}
        self.alpha = 0.85
        self.boot_time = time.time()

    def draw_glitch_text(self, img, text, pos, color):
        offset = np.random.randint(-1, 2)
        cv2.putText(img, text, (pos[0] + offset, pos[1] + offset),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1, cv2.LINE_AA)

    def draw_skeleton_pro(self, frame, results):
        if not results or results[0].keypoints is None: return frame

        # Kadang data di GPU harus dipindah ke CPU untuk operasi NumPy
        kpts_data = results[0].keypoints.data.cpu().numpy()
        boxes = results[0].boxes
        ids = boxes.id.int().cpu().numpy() if boxes.id is not None else range(len(kpts_data))

        self.trail_buffer.append(kpts_data.copy())

        for idx, person in enumerate(kpts_data):
            pid = ids[idx]
            conf_avg = np.mean(person[:, 2])

            # Trail Ghosting
            for past_frame in self.trail_buffer:
                if idx < len(past_frame):
                    for p1, p2 in [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (12, 14)]:
                        pt1 = (int(past_frame[idx][p1][0]), int(past_frame[idx][p1][1]))
                        pt2 = (int(past_frame[idx][p2][0]), int(past_frame[idx][p2][1]))
                        cv2.line(frame, pt1, pt2, (0, 70, 0), 1)

            # Bounding Box
            bbox = boxes.xyxy[idx].cpu().numpy()
            x1, y1, x2, y2 = map(int, bbox)
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            connections = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                           (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

            thickness = max(1, int(3 - (y2 - y1) / 300))

            for p1, p2 in connections:
                if person[p1][2] > 0.5 and person[p2][2] > 0.5:
                    c1, c2 = (int(person[p1][0]), int(person[p1][1])), (int(person[p2][0]), int(person[p2][1]))
                    cv2.line(frame, c1, c2, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.line(frame, c1, c2, (0, 255, 127), thickness + 1, cv2.LINE_AA)

            for px, py, pconf in person:
                if pconf > 0.6:
                    radius = int(3 + math.sin(time.time() * 10) * 2)
                    cv2.circle(frame, (int(px), int(py)), radius, (0, 255, 0), 1)

        return frame

    def apply_cyber_filter(self, frame):
        frame[:, :, 0] = frame[:, :, 0] * 0.1
        frame[:, :, 2] = frame[:, :, 2] * 0.1
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            # --- INFERENCE ON DEVICE ---
            results = self.model.track(frame, persist=True, verbose=False, conf=0.4, device=self.device)

            canvas = self.apply_cyber_filter(frame.copy())
            canvas = cv2.addWeighted(canvas, 0.4, np.zeros_like(frame), 0.6, 0)
            canvas = self.draw_skeleton_pro(canvas, results)

            # HUD Status
            cv2.putText(canvas, f"SYSTEM: {self.device}", (20, 30), 0, 0.6, (255, 255, 255), 1)
            cv2.imshow("GOD-EYE V8.2", canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    GodEyeV8().run()