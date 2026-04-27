import cv2
import numpy as np
from ultralytics import YOLO
import time

# ==========================================
# KONFIGURASI SISTEM
# ==========================================
# Model yang digunakan (Nano segmentation untuk kecepatan real-time)
MODEL_PATH = "yolov8n-seg.pt"
# Mode Awal: 'thermal' atau 'normal'
INITIAL_MODE = 'thermal'
# Status Toggles Awal
STATUS_NIGHT_VISION = True
STATUS_THERMAL_MODE = True  # Jika mode awal thermal, ini harus True

# Warna Tampilan (BGR)
CLR_TACTICAL_GREEN = (0, 255, 0)
CLR_DANGER_RED = (0, 0, 255)
CLR_WHITE = (255, 255, 255)
CLR_BLACK = (0, 0, 0)


# ==========================================
# CLASS VISI TAKTIS
# ==========================================
class TacticalScope:
    def __init__(self):
        # Load Model
        print(f"⏳ Loading model: {MODEL_PATH}...")
        self.model = YOLO(MODEL_PATH)
        print("✅ Model Loaded.")

        # System State
        self.night_vision = STATUS_NIGHT_VISION
        self.thermal_mode = STATUS_THERMAL_MODE
        self.target_locked = False

        # FPS Counter
        self.prev_frame_time = 0
        self.new_frame_time = 0

    def apply_thermal_visualization(self, frame):
        """Mengubah frame normal menjadi visualisasi panas."""
        # Ubah ke Grayscale terlebih dahulu
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Terapkan Colormap JET (Biru=Dingin, Merah=Panas)
        thermal_frame = cv2.applyColorMap(gray_frame, cv2.COLORMAP_JET)
        return thermal_frame

    def apply_night_vision_effect(self, frame):
        """Terapkan efek visi malam hijau taktis."""
        # Split channel BGR
        b, g, r = cv2.split(frame)
        # Buat channel kosong
        blank = np.zeros_like(g)
        # Gabungkan kembali dengan channel hijau sebagai dominan, lainnya kosong
        # Ini memberikan tint hijau monokromatik
        night_vision_frame = cv2.merge([blank, g, blank])
        return night_vision_frame

    def draw_tactical_hud(self, frame, results):
        """Gambar elemen UI Taktis (HUD)."""
        h, w, _ = frame.shape

        # --- 1. System Info Bar (Atas) ---
        hud_top = frame.copy()
        cv2.rectangle(hud_top, (0, 0), (w, 60), CLR_BLACK, -1)
        cv2.addWeighted(hud_top, 0.5, frame, 0.5, 0, frame)

        # GPU Status (Dummy), System Name, Mode, Toggles
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "GPU: NVIDIA RTX | CUDA: ACTIVE", (20, 35), font, 0.7, CLR_WHITE, 2)
        cv2.putText(frame, "SYSTEM: TACTICAL_OS_V3", (w - 350, 35), font, 0.7, CLR_TACTICAL_GREEN, 2)

        # --- 2. Toggles Tampilan (Kanan) ---
        toggle_box_w = 250
        toggle_box_h = 150
        toggle_box_x = w - toggle_box_w - 20
        toggle_box_y = 80

        # Latar belakang Toggle
        toggle_bg = frame.copy()
        cv2.rectangle(toggle_bg, (toggle_box_x, toggle_box_y), (w - 20, toggle_box_y + toggle_box_h), CLR_BLACK, -1)
        cv2.addWeighted(toggle_bg, 0.4, frame, 0.6, 0, frame)

        # Fungsi Gambar Toggle
        def draw_toggle(img, y_pos, text, state):
            color = CLR_TACTICAL_GREEN if state else (100, 100, 100)
            status_txt = "ON" if state else "OFF"
            cv2.putText(img, f"{text}:", (toggle_box_x + 15, y_pos), font, 0.6, CLR_WHITE, 1)
            cv2.putText(img, status_txt, (toggle_box_x + 180, y_pos), font, 0.6, color, 2)

        draw_toggle(frame, toggle_box_y + 40, "THERMAL_MODE", self.thermal_mode)
        draw_toggle(frame, toggle_box_y + 90, "NIGHT_VISION", self.night_vision)
        cv2.putText(frame, "('T' to Toggle)", (toggle_box_x + 15, toggle_box_y + 130), font, 0.5, (150, 150, 150), 1)

        # --- 3. FPS Counter & Status Taktis ---
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, h - 30), font, 0.6, CLR_WHITE, 1)

        status_color = CLR_DANGER_RED if self.target_locked else CLR_TACTICAL_GREEN
        status_txt = "!!! HAZARD DETECTED !!!" if self.target_locked else "SCANNING..."
        cv2.putText(frame, status_txt, (w // 2 - 150, h - 30), font, 0.8, status_color, 2)

        # --- 4. Objek & Segmentation (Jika Termal Aktif) ---
        if self.thermal_mode:
            # results[0].plot() akan menggambar BB, Labels, dan Mask asli YOLO
            # Namun kita ingin menggambarnya di atas frame TERMAL
            # Jadi kita perlu teknik blend mask

            # Buat frame kosong untuk menggambar mask
            mask_frame = np.zeros_like(frame)

            # Ambil deteksi orang (Class 0 di COCO)
            self.target_locked = False
            for result in results:
                masks = result.masks
                boxes = result.boxes

                if masks is not None:
                    for i, (mask, box) in enumerate(zip(masks.xy, boxes)):
                        class_id = int(box.cls[0])

                        # Hanya proses 'person'
                        if class_id == 0:
                            self.target_locked = True

                            # Gambar Mask Hijau di frame kosong
                            polygon = mask.astype(np.int32)
                            cv2.fillPoly(mask_frame, [polygon], CLR_TACTICAL_GREEN)

                            # Gambar Bounding Box Kuning
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                            # Label Info
                            cv2.putText(frame, f"USER_ID: 00{i + 1}", (x1, y1 - 10), font, 0.5, (0, 255, 255), 1)

            # Blend Mask Hijau ke Frame Termal
            # Mask Hijau hanya muncul di area objek 'person'
            frame = cv2.addWeighted(frame, 1.0, mask_frame, 0.3, 0)

        else:
            # Jika Normal/Night Vision, gunakan plot standar YOLO
            frame = results[0].plot(boxes=True, masks=True, labels=True)

        return frame

    def process_video(self):
        cap = cv2.VideoCapture(0)  # Gunakan Webcam

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)  # Flip horizontal

            # Update FPS Time
            self.new_frame_time = time.time()

            # --- PRE-PROCESSING VIEW MODE ---
            base_view = frame.copy()

            # 1. Terapkan Mode Thermal
            if self.thermal_mode:
                base_view = self.apply_thermal_visualization(base_view)

            # 2. Terapkan Mode Night Vision (Overlay Tint Hijau)
            if self.night_vision:
                base_view = self.apply_night_vision_effect(base_view)

            # --- INFERENCE (YOLO) ---
            # Jalankan deteksi di atas frame asli atau yang sudah dimodifikasi
            # Untuk akurasi segmentasi terbaik, gunakan frame normal untuk inference
            # Namun untuk visualisasi, kita tampilkan hasil di view mode
            yolo_results = self.model.track(frame, persist=True, verbose=False, conf=0.5, classes=0)

            # --- DRAW HUD & RESULTS ---
            tactical_view = self.draw_tactical_hud(base_view, yolo_results)

            # Update FPS Time
            self.prev_frame_time = self.new_frame_time

            # Tampilkan Hasil
            cv2.imshow("Nuclear Tactical Scope v3.0", tactical_view)

            # --- KEYBOARD CONTROLS ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 'Q' untuk keluar
                break
            elif key == ord('t'):  # 'T' untuk toggle mode views
                # Cycle through: Normal -> Night Vision -> Thermal+Night Vision -> Thermal -> Normal
                if not self.thermal_mode and not self.night_vision:
                    self.night_vision = True
                elif not self.thermal_mode and self.night_vision:
                    self.thermal_mode = True
                elif self.thermal_mode and self.night_vision:
                    self.night_vision = False
                elif self.thermal_mode and not self.night_vision:
                    self.thermal_mode = False
                    self.night_vision = False

        cap.release()
        cv2.destroyAllWindows()


# ==========================================
# RUN SYSTEM
# ==========================================
if __name__ == "__main__":
    scope = TacticalScope()
    scope.process_video()