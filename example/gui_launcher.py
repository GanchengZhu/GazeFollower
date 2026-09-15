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
        self.backend_var = tk.StringVar(value="PyGame")
        self.use_mp_var = tk.BooleanVar(value=True)
        self.enable_face_filter_var = tk.BooleanVar(value=True)
        
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
        ttk.Checkbutton(face_frame, text="Enable 1-Euro Filter (reduces landmark & eye jitter)", variable=self.enable_face_filter_var).pack(side=tk.LEFT, padx=15)
        
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

        # 6. UI Backend & Process Isolation Settings
        backend_frame = ttk.LabelFrame(main_frame, text="6. UI Backend & Process Isolation", padding="10")
        backend_frame.pack(fill=tk.X, pady=4)

        backend_row = ttk.Frame(backend_frame)
        backend_row.pack(fill=tk.X, pady=2)
        ttk.Label(backend_row, text="UI Backend:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(backend_row, text="PyGame", variable=self.backend_var, value="PyGame").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(backend_row, text="PsychoPy (Scientific)", variable=self.backend_var, value="PsychoPy").pack(side=tk.LEFT, padx=10)

        mp_row = ttk.Frame(backend_frame)
        mp_row.pack(fill=tk.X, pady=4)
        self.mp_cb = ttk.Checkbutton(
            mp_row,
            text="Enable Multiprocessing (Isolates eye-tracking ML to separate process; recommended for PsychoPy)",
            variable=self.use_mp_var
        )
        self.mp_cb.pack(side=tk.LEFT, padx=5)
        
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
        
        use_mp = bool(self.use_mp_var.get())
        backend_choice = self.backend_var.get()
        enable_face_filter = bool(self.enable_face_filter_var.get())
        print(f" - UI Backend: {backend_choice}")
        print(f" - Multiprocessing: {use_mp}")
        print(f" - Face 1-Euro Filter: {enable_face_filter}")
        
        # Initialize components based on selection
        if self.face_align_var.get() == "BlazeFace":
            face_alignment = BlazeFaceAlignment(enable_filter=enable_face_filter)
        else:
            face_alignment = MediaPipeFaceAlignment(enable_filter=enable_face_filter)
            
        gaze_estimator = MGazeNetGazeEstimator(model_path=model_path)
        
        if self.calib_var.get() == "MultivariateRidge":
            calibration = MultivariateRidgeCalibration(alpha=0.1)
        else:
            calibration = SVRCalibration()
            
        config = DefaultConfig()
        config.use_multiprocessing = use_mp
        config.enable_face_filter = enable_face_filter
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
            config=config,
            use_multiprocessing=use_mp
        )
        
        screen_w = int(config.screen_size[0])
        screen_h = int(config.screen_size[1])

        if backend_choice == "PsychoPy":
            from psychopy import visual, event
            win = visual.Window(
                size=(screen_w, screen_h),
                fullscr=True,
                units='pix',
                color=[128, 128, 128],
                colorSpace='rgb255'
            )
            
            gf.preview(win=win)
            gf.calibrate(win=win)
            gf.start_sampling()
            
            text_stim = visual.TextStim(
                win,
                text="Look around and press SPACE or ESC to exit",
                color=[255, 255, 255],
                colorSpace='rgb255',
                height=36,
                units='pix',
                pos=(0, 0)
            )
            gaze_circle = visual.ShapeStim(
                win,
                vertices='circle',
                size=(50, 50),
                fillColor=None,
                lineColor=[0, 255, 0],
                lineWidth=4,
                colorSpace='rgb255',
                units='pix'
            )
            
            running = True
            while running:
                keys = event.getKeys()
                if 'space' in keys or 'escape' in keys:
                    running = False
                
                text_stim.draw()
                gaze_info = gf.get_gaze_info()
                if gaze_info and gaze_info.status and gaze_info.filtered_gaze_coordinates is not None:
                    gx, gy = gaze_info.filtered_gaze_coordinates
                    p_x = gx - screen_w // 2
                    p_y = -(gy - screen_h // 2)
                    gaze_circle.pos = (p_x, p_y)
                    gaze_circle.draw()
                    
                win.flip()
                
            gf.stop_sampling()
            gf.release()
            win.close()
        else:
            # Start the Pygame experiment
            pygame.init()
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
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT or (ev.type == KEYDOWN and ev.key in (K_SPACE, pygame.K_ESCAPE)):
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
