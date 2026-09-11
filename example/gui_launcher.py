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

class LauncherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GazeFollower SDK - Test Launcher")
        self.root.geometry("1000x450")
        
        # Variables
        self.model_path_var = tk.StringVar(value="base.mnn")
        self.face_align_var = tk.StringVar(value="BlazeFace")
        self.calib_var = tk.StringVar(value="MultivariateRidge")
        
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Configure GazeFollower Session", font=("Segoe UI", 14, "bold")).pack(pady=(0, 20))
        
        # Model Selection
        model_frame = ttk.LabelFrame(main_frame, text="1. Gaze Estimator Model", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_path_var, values=["base.mnn", "mobilenet_v4.mnn"], state="readonly", width=30)
        model_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Browse...", command=self.browse_model).pack(side=tk.LEFT, padx=5)
        
        # Face Alignment Selection
        face_frame = ttk.LabelFrame(main_frame, text="2. Face Alignment", padding="10")
        face_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(face_frame, text="BlazeFace (Fastest)", variable=self.face_align_var, value="BlazeFace").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(face_frame, text="MediaPipe (High Precision)", variable=self.face_align_var, value="MediaPipe").pack(side=tk.LEFT, padx=10)
        
        # Calibration Selection
        calib_frame = ttk.LabelFrame(main_frame, text="3. Calibration Algorithm", padding="10")
        calib_frame.pack(fill=tk.X, pady=5)

        ttk.Radiobutton(calib_frame, text="Multivariate Ridge", variable=self.calib_var, value="MultivariateRidge").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(calib_frame, text="SVR (Support Vector)", variable=self.calib_var, value="SVR").pack(side=tk.LEFT, padx=10)
        
        # Launch Button
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=20)
        ttk.Button(btn_frame, text="Launch Experiment", command=self.launch, style="Accent.TButton").pack(side=tk.RIGHT)
        
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
        print(f" - Calibration: {self.calib_var.get()}")
        
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
            
        gf = GazeFollower(
            face_alignment=face_alignment,
            gaze_estimator=gaze_estimator,
            calibration=calibration
        )
        
        # Start the Pygame experiment
        pygame.init()
        win = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
        
        gf.preview(win=win)
        gf.calibrate(win=win)
        gf.start_sampling()
        pygame.time.wait(100)
        
        win.fill((128, 128, 128))
        font = pygame.font.Font(None, 74)
        text = font.render("Look around and press SPACE to exit", True, (255, 255, 255))
        win.blit(text, (300, 500))
        pygame.display.flip()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == KEYDOWN and event.key == K_SPACE:
                    running = False
            
            gaze_info = gf.get_gaze_info()
            if gaze_info and gaze_info.status:
                win.fill((128, 128, 128))
                win.blit(text, (300, 500))
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
