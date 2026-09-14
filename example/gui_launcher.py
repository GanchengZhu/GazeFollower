import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pygame
from pathlib import Path

from pygame import KEYDOWN, K_SPACE

# Add the parent directory to PYTHONPATH if we want to run locally
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gazefollower import GazeFollower
from gazefollower.face_alignment import BlazeFaceAlignment, MediaPipeFaceAlignment
from gazefollower.gaze_estimator import MGazeNetGazeEstimator
from gazefollower.calibration import MultivariateRidgeCalibration, SVRCalibration
from gazefollower.misc import DefaultConfig, CalibrationMode

class LauncherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GazeFollower SDK - Test Launcher")
        self.root.geometry("850x640")
        self.root.minsize(750, 580)
        
        # Variables
        self.model_path_var = tk.StringVar(value="base.mnn")
        self.face_align_var = tk.StringVar(value="BlazeFace")
        self.calib_var = tk.StringVar(value="MultivariateRidge")
        self.cali_mode_var = tk.StringVar(value="9")
        self.cali_click_mode_var = tk.BooleanVar(value=False)
        self.lissajous_latency_var = tk.IntVar(value=4)
        
        self.create_widgets()
        self.on_cali_mode_changed()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Configure GazeFollower Session", font=("Segoe UI", 14, "bold")).pack(pady=(0, 15))
        
        # 1. Model Selection
        model_frame = ttk.LabelFrame(main_frame, text="1. Gaze Estimator Model", padding="10")
        model_frame.pack(fill=tk.X, pady=4)
        
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_path_var, values=["base.mnn", "mobilenet_v4.mnn"], state="readonly", width=30)
        model_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Browse...", command=self.browse_model).pack(side=tk.LEFT, padx=5)
        
        # 2. Face Alignment Selection
        face_frame = ttk.LabelFrame(main_frame, text="2. Face Alignment", padding="10")
        face_frame.pack(fill=tk.X, pady=4)
        
        ttk.Radiobutton(face_frame, text="BlazeFace (Fastest)", variable=self.face_align_var, value="BlazeFace").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(face_frame, text="MediaPipe (High Precision)", variable=self.face_align_var, value="MediaPipe").pack(side=tk.LEFT, padx=10)
        
        # 3. Calibration Algorithm Selection
        calib_frame = ttk.LabelFrame(main_frame, text="3. Calibration Algorithm", padding="10")
        calib_frame.pack(fill=tk.X, pady=4)

        ttk.Radiobutton(calib_frame, text="Multivariate Ridge", variable=self.calib_var, value="MultivariateRidge").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(calib_frame, text="SVR (Support Vector)", variable=self.calib_var, value="SVR").pack(side=tk.LEFT, padx=10)
        
        # 4. Calibration Pattern & Mode Selection
        pattern_frame = ttk.LabelFrame(main_frame, text="4. Calibration Pattern", padding="10")
        pattern_frame.pack(fill=tk.X, pady=4)
        
        ttk.Radiobutton(pattern_frame, text="9-Point (Default)", variable=self.cali_mode_var, value="9", command=self.on_cali_mode_changed).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(pattern_frame, text="5-Point (Fast)", variable=self.cali_mode_var, value="5", command=self.on_cali_mode_changed).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(pattern_frame, text="13-Point (High Precision)", variable=self.cali_mode_var, value="13", command=self.on_cali_mode_changed).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(pattern_frame, text="Lissajous (Smooth Pursuit)", variable=self.cali_mode_var, value="Lissajous", command=self.on_cali_mode_changed).pack(side=tk.LEFT, padx=10)
        
        # 5. Calibration Interaction & Latency Settings
        opt_frame = ttk.LabelFrame(main_frame, text="5. Calibration Options", padding="10")
        opt_frame.pack(fill=tk.X, pady=4)
        
        # Scheme A: Click mode
        click_row = ttk.Frame(opt_frame)
        click_row.pack(fill=tk.X, pady=2)
        self.click_cb = ttk.Checkbutton(
            click_row,
            text="Point-and-Click Calibration (Scheme A: user clicks targets)",
            variable=self.cali_click_mode_var
        )
        self.click_cb.pack(side=tk.LEFT, padx=5)
        self.click_hint = ttk.Label(click_row, text="(Disabled in Lissajous mode)", font=("Segoe UI", 9, "italic"), foreground="gray")
        self.click_hint.pack(side=tk.LEFT, padx=5)

        # Lissajous Latency setting
        latency_row = ttk.Frame(opt_frame)
        latency_row.pack(fill=tk.X, pady=4)
        self.latency_label = ttk.Label(latency_row, text="Lissajous Camera Latency (frames):")
        self.latency_label.pack(side=tk.LEFT, padx=5)
        self.latency_spin = ttk.Spinbox(latency_row, from_=0, to=16, textvariable=self.lissajous_latency_var, width=5)
        self.latency_spin.pack(side=tk.LEFT, padx=5)
        self.latency_desc = ttk.Label(latency_row, text="(Compensates for camera ISP & frame buffer delay, default: 4)", font=("Segoe UI", 9, "italic"))
        self.latency_desc.pack(side=tk.LEFT, padx=5)
        
        # Launch Button
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=15)
        ttk.Button(btn_frame, text="Launch Experiment", command=self.launch, style="Accent.TButton").pack(side=tk.RIGHT)

    def on_cali_mode_changed(self):
        mode = self.cali_mode_var.get()
        if mode == "Lissajous":
            # Lissajous calibration is continuous gaze pursuit only (no click calibration)
            self.cali_click_mode_var.set(False)
            self.click_cb.config(state="disabled")
            self.click_hint.config(foreground="gray")
            self.latency_label.config(state="normal")
            self.latency_spin.config(state="normal")
            self.latency_desc.config(state="normal")
        else:
            self.click_cb.config(state="normal")
            self.click_hint.config(foreground="#666666")
            self.latency_label.config(state="disabled")
            self.latency_spin.config(state="disabled")
            self.latency_desc.config(state="disabled")
            
    def browse_model(self):
        path = filedialog.askopenfilename(filetypes=[("MNN Models", "*.mnn"), ("All Files", "*.*")])
        if path:
            self.model_path_var.set(path)
            
    def launch(self):
        self.root.destroy()
        
        import gazefollower
        res_dir = Path(gazefollower.__file__).parent / "res" / "model_weights"
        
        model_val = self.model_path_var.get()
        if model_val in ["base.mnn", "mobilenet_v4.mnn"]:
            model_path = str(res_dir / model_val)
        else:
            model_path = model_val
            
        print(f"Launching with:")
        print(f" - Model: {model_path}")
        print(f" - Face Alignment: {self.face_align_var.get()}")
        print(f" - Calibration Algorithm: {self.calib_var.get()}")
        print(f" - Calibration Pattern: {self.cali_mode_var.get()}")
        print(f" - Click Calibration: {self.cali_click_mode_var.get()}")
        if self.cali_mode_var.get() == "Lissajous":
            print(f" - Lissajous Latency: {self.lissajous_latency_var.get()} frames")
        
        # Initialize components based on selection
        if self.face_align_var.get() == "BlazeFace":
            face_alignment = BlazeFaceAlignment()
        else:
            face_alignment = MediaPipeFaceAlignment()
            
        gaze_estimator = MGazeNetGazeEstimator(model_path=model_path)
        
        if self.calib_var.get() == "MultivariateRidge":
            calibration = MultivariateRidgeCalibration(alpha=0.1)
        else:
            calibration = SVRCalibration()
            
        config = DefaultConfig()
        mode_val = self.cali_mode_var.get()
        if mode_val == "Lissajous":
            config.cali_mode = CalibrationMode.LISSAJOUS
        elif mode_val == "5":
            config.cali_mode = CalibrationMode.FIVE_POINT
        elif mode_val == "9":
            config.cali_mode = CalibrationMode.NINE_POINT
        elif mode_val == "13":
            config.cali_mode = CalibrationMode.THIRTEEN_POINT
            
        config.cali_click_mode = bool(self.cali_click_mode_var.get())
        try:
            config.lissajous_frame_latency = int(self.lissajous_latency_var.get())
        except (ValueError, tk.TclError):
            config.lissajous_frame_latency = 4
            
        gf = GazeFollower(
            face_alignment=face_alignment,
            gaze_estimator=gaze_estimator,
            calibration=calibration,
            config=config
        )
        
        # Start the Pygame experiment
        pygame.init()
        screen_w = int(config.screen_size[0])
        screen_h = int(config.screen_size[1])
        win = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
        
        gf.preview(win=win)
        gf.calibrate(win=win)
        gf.start_sampling()
        pygame.time.wait(100)
        
        win.fill((128, 128, 128))
        font = pygame.font.Font(None, 74)
        text = font.render("Look around and press SPACE or ESC to exit", True, (255, 255, 255))
        win.blit(text, (screen_w // 2 - text.get_width() // 2, screen_h // 2 - text.get_height() // 2))
        pygame.display.flip()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key in (K_SPACE, pygame.K_ESCAPE)):
                    running = False
            
            gaze_info = gf.get_gaze_info()
            if gaze_info and gaze_info.status:
                win.fill((128, 128, 128))
                win.blit(text, (screen_w // 2 - text.get_width() // 2, screen_h // 2 - text.get_height() // 2))
                gx = int(gaze_info.filtered_gaze_coordinates[0])
                gy = int(gaze_info.filtered_gaze_coordinates[1])
                pygame.draw.circle(win, (0, 255, 0), (gx, gy), 50, 5)
                pygame.display.flip()
                
            pygame.time.wait(10)
            
        gf.stop_sampling()
        gf.release()
        pygame.quit()

if __name__ == "__main__":
    root = tk.Tk()
    # Simple accent style for launch button
    style = ttk.Style()
    style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
    app = LauncherGUI(root)
    root.mainloop()
