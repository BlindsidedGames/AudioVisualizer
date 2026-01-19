"""GUI interface for the visualizer tool using CustomTkinter."""

import customtkinter as ctk
from tkinter import filedialog
from pathlib import Path
from threading import Thread
import sys
import os
import time

from PIL import Image
import numpy as np


def get_app_path():
    """Get the application root path (works for both dev and bundled exe)."""
    if getattr(sys, 'frozen', False):
        # Running as bundled exe
        return Path(sys._MEIPASS)
    else:
        # Running as script
        return Path(__file__).parent.parent


def get_output_path():
    """Get the output directory path."""
    if getattr(sys, 'frozen', False):
        # When bundled, use exe directory for outputs
        return Path(sys.executable).parent / "outputs"
    else:
        return Path(__file__).parent.parent / "outputs"


class VisualizerApp(ctk.CTk):
    """Main application window."""

    def __init__(self):
        super().__init__()

        self.title("Audio Visualizer")
        self.geometry("920x850")
        self.resizable(False, False)

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # State
        self.audio_path = None
        self.output_dir = get_output_path()
        self.rendering = False
        self.stop_requested = False
        self.render_thread = None

        # Preview state
        self.preview_running = False
        self.preview_thread = None
        self.preview_renderer = None
        self.preview_stop_requested = False

        # Detect capabilities
        self.capabilities = self._detect_capabilities()

        # Build UI
        self._create_widgets()

    def _detect_capabilities(self) -> dict:
        """Detect available rendering backends."""
        caps = {
            "cuda": False,
            "cupy": False,
            "nvenc": False,
            "opengl": False,
            "bundled": getattr(sys, 'frozen', False),  # Running as exe?
        }

        # Check OpenGL (should always work)
        try:
            import moderngl
            caps["opengl"] = True
        except ImportError:
            pass

        # Check CUDA via CuPy
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            caps["cupy"] = True
            caps["cuda"] = True
        except:
            pass

        # Check CUDA via PyTorch (fallback)
        if not caps["cuda"]:
            try:
                import torch
                if torch.cuda.is_available():
                    caps["cuda"] = True
            except:
                pass

        # Check NVENC (via FFmpeg) - use bundled ffmpeg if available
        try:
            import subprocess
            import shutil

            # Check bundled ffmpeg first
            if caps["bundled"]:
                ffmpeg = str(Path(sys.executable).parent / "ffmpeg" / "ffmpeg.exe")
                if not Path(ffmpeg).exists():
                    ffmpeg = shutil.which("ffmpeg")
            else:
                ffmpeg = shutil.which("ffmpeg")

            if ffmpeg:
                result = subprocess.run(
                    [ffmpeg, "-encoders"],
                    capture_output=True, text=True, timeout=5
                )
                if "h264_nvenc" in result.stdout:
                    caps["nvenc"] = True
        except:
            pass

        return caps

    def _create_widgets(self):
        """Create all UI widgets in two-column layout."""
        # Main container with padding
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Two-column layout
        left_column = ctk.CTkFrame(main_container, fg_color="transparent", width=480)
        left_column.pack(side="left", fill="y", padx=(0, 10))
        left_column.pack_propagate(False)

        right_column = ctk.CTkFrame(main_container, fg_color="transparent", width=390)
        right_column.pack(side="right", fill="y")
        right_column.pack_propagate(False)

        # ========== LEFT COLUMN (existing controls) ==========
        container = left_column

        # Title
        title = ctk.CTkLabel(
            container,
            text="Audio Visualizer",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 5))

        # Capability status
        status_text = self._get_status_text()
        self.status_label = ctk.CTkLabel(
            container,
            text=status_text,
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.status_label.pack(pady=(0, 15))

        # Audio file section
        audio_frame = ctk.CTkFrame(container)
        audio_frame.pack(fill="x", pady=(0, 15))

        audio_label = ctk.CTkLabel(
            audio_frame,
            text="Audio File",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        audio_label.pack(anchor="w", padx=15, pady=(15, 5))

        file_row = ctk.CTkFrame(audio_frame, fg_color="transparent")
        file_row.pack(fill="x", padx=15, pady=(0, 15))

        self.file_entry = ctk.CTkEntry(
            file_row,
            placeholder_text="Drop audio file here or click Browse...",
            state="disabled",
            width=300
        )
        self.file_entry.pack(side="left", fill="x", expand=True)

        browse_btn = ctk.CTkButton(
            file_row,
            text="Browse",
            command=self._browse_file,
            width=80
        )
        browse_btn.pack(side="right", padx=(10, 0))

        # Resolution section
        res_frame = ctk.CTkFrame(container)
        res_frame.pack(fill="x", pady=(0, 15))

        res_label = ctk.CTkLabel(
            res_frame,
            text="Resolution",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        res_label.pack(anchor="w", padx=15, pady=(15, 5))

        # Preset dropdown
        preset_row = ctk.CTkFrame(res_frame, fg_color="transparent")
        preset_row.pack(fill="x", padx=15, pady=(0, 10))

        self.res_preset = ctk.CTkComboBox(
            preset_row,
            values=[
                "1920x1080 (1080p)",
                "1280x720 (720p)",
                "3840x2160 (4K)",
                "720x1280 (Vertical 720p)",
                "1080x1920 (Vertical 1080p)",
                "Custom"
            ],
            command=self._on_preset_change,
            width=200
        )
        self.res_preset.set("1920x1080 (1080p)")
        self.res_preset.pack(side="left")

        # Custom resolution inputs
        custom_row = ctk.CTkFrame(res_frame, fg_color="transparent")
        custom_row.pack(fill="x", padx=15, pady=(0, 15))

        width_label = ctk.CTkLabel(custom_row, text="Width:")
        width_label.pack(side="left")

        self.width_entry = ctk.CTkEntry(custom_row, width=70)
        self.width_entry.insert(0, "1920")
        self.width_entry.pack(side="left", padx=(5, 15))

        height_label = ctk.CTkLabel(custom_row, text="Height:")
        height_label.pack(side="left")

        self.height_entry = ctk.CTkEntry(custom_row, width=70)
        self.height_entry.insert(0, "1080")
        self.height_entry.pack(side="left", padx=(5, 0))

        # Shader preset section
        shader_frame = ctk.CTkFrame(container)
        shader_frame.pack(fill="x", pady=(0, 15))

        shader_label = ctk.CTkLabel(
            shader_frame,
            text="Visual Preset",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        shader_label.pack(anchor="w", padx=15, pady=(15, 5))

        shader_row = ctk.CTkFrame(shader_frame, fg_color="transparent")
        shader_row.pack(fill="x", padx=15, pady=(0, 15))

        # Find available shaders
        shaders = self._get_shader_list()

        self.shader_combo = ctk.CTkComboBox(
            shader_row,
            values=shaders,
            width=200,
            command=self._on_shader_change
        )
        self.shader_combo.set(shaders[0] if shaders else "universe_within")
        self.shader_combo.pack(side="left")

        refresh_btn = ctk.CTkButton(
            shader_row,
            text="Refresh",
            command=self._refresh_shaders,
            width=70
        )
        refresh_btn.pack(side="left", padx=(10, 0))

        # Zoom section
        zoom_row = ctk.CTkFrame(shader_frame, fg_color="transparent")
        zoom_row.pack(fill="x", padx=15, pady=(0, 15))

        zoom_label = ctk.CTkLabel(zoom_row, text="Zoom:")
        zoom_label.pack(side="left")

        self.zoom_var = ctk.DoubleVar(value=1.0)
        self.zoom_slider = ctk.CTkSlider(
            zoom_row,
            from_=0.5,
            to=4.0,
            number_of_steps=35,  # 0.1 increments
            variable=self.zoom_var,
            command=self._on_zoom_change,
            width=150
        )
        self.zoom_slider.pack(side="left", padx=(10, 10))

        self.zoom_value_label = ctk.CTkLabel(zoom_row, text="1.0x", width=40)
        self.zoom_value_label.pack(side="left")

        # Duration section (for preview/testing)
        duration_frame = ctk.CTkFrame(container)
        duration_frame.pack(fill="x", pady=(0, 15))

        duration_label = ctk.CTkLabel(
            duration_frame,
            text="Duration",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        duration_label.pack(anchor="w", padx=15, pady=(15, 5))

        duration_row = ctk.CTkFrame(duration_frame, fg_color="transparent")
        duration_row.pack(fill="x", padx=15, pady=(0, 15))

        self.duration_var = ctk.StringVar(value="full")

        full_radio = ctk.CTkRadioButton(
            duration_row,
            text="Full song",
            variable=self.duration_var,
            value="full"
        )
        full_radio.pack(side="left", padx=(0, 15))

        preview_radio = ctk.CTkRadioButton(
            duration_row,
            text="Preview:",
            variable=self.duration_var,
            value="preview"
        )
        preview_radio.pack(side="left")

        self.preview_seconds = ctk.CTkEntry(
            duration_row,
            width=50,
            placeholder_text="10"
        )
        self.preview_seconds.insert(0, "10")
        self.preview_seconds.pack(side="left", padx=(5, 0))

        seconds_label = ctk.CTkLabel(duration_row, text="seconds")
        seconds_label.pack(side="left", padx=(5, 0))

        # Output folder section
        output_frame = ctk.CTkFrame(container)
        output_frame.pack(fill="x", pady=(0, 15))

        output_label = ctk.CTkLabel(
            output_frame,
            text="Output Folder",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        output_label.pack(anchor="w", padx=15, pady=(15, 5))

        output_row = ctk.CTkFrame(output_frame, fg_color="transparent")
        output_row.pack(fill="x", padx=15, pady=(0, 15))

        self.output_entry = ctk.CTkEntry(
            output_row,
            state="disabled",
            width=300
        )
        self.output_entry.pack(side="left", fill="x", expand=True)
        self._update_output_display()

        output_btn = ctk.CTkButton(
            output_row,
            text="Change",
            command=self._browse_output,
            width=80
        )
        output_btn.pack(side="right", padx=(10, 0))

        # Progress section
        progress_frame = ctk.CTkFrame(container)
        progress_frame.pack(fill="x", pady=(0, 15))

        self.progress_label = ctk.CTkLabel(
            progress_frame,
            text="Ready",
            font=ctk.CTkFont(size=12)
        )
        self.progress_label.pack(anchor="w", padx=15, pady=(15, 5))

        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=400)
        self.progress_bar.pack(padx=15, pady=(0, 15))
        self.progress_bar.set(0)

        # Render button
        self.render_btn = ctk.CTkButton(
            container,
            text="Start Render",
            command=self._start_render,
            height=45,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.render_btn.pack(fill="x", pady=(10, 0))

        # Enable drag and drop
        self._setup_drag_drop()

        # ========== RIGHT COLUMN (preview and sliders) ==========
        self._create_preview_panel(right_column)
        self._create_parameter_sliders(right_column)

    def _get_status_text(self) -> str:
        """Get status text based on capabilities."""
        if self.capabilities["nvenc"]:
            return "NVENC Encoding"
        return ""

    def _get_shader_list(self) -> list:
        """Get list of available shaders."""
        shader_dir = get_app_path() / "shaders"
        if shader_dir.exists():
            return sorted([f.stem for f in shader_dir.glob("*.glsl")])
        return ["universe_within"]

    def _refresh_shaders(self):
        """Refresh the shader list."""
        current = self.shader_combo.get()
        shaders = self._get_shader_list()
        self.shader_combo.configure(values=shaders)
        # Keep current selection if it still exists
        if current in shaders:
            self.shader_combo.set(current)
        elif shaders:
            self.shader_combo.set(shaders[0])

    def _on_zoom_change(self, value):
        """Update zoom label when slider changes."""
        self.zoom_value_label.configure(text=f"{value:.1f}x")

    def _on_shader_change(self, value):
        """Handle shader selection change - restart preview if running."""
        if self.preview_running:
            # Stop and restart preview with new shader
            self._stop_preview()
            # Wait a bit for the preview to stop, then restart
            self.after(200, self._start_preview)

    def _create_preview_panel(self, parent):
        """Create the shader preview panel."""
        # Preview frame
        preview_frame = ctk.CTkFrame(parent)
        preview_frame.pack(fill="x", pady=(0, 15))

        preview_label = ctk.CTkLabel(
            preview_frame,
            text="Shader Preview",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        preview_label.pack(anchor="w", padx=15, pady=(15, 5))

        # Preview canvas (16:9 aspect ratio, 380x214)
        preview_container = ctk.CTkFrame(preview_frame, fg_color="#1a1a1a")
        preview_container.pack(padx=15, pady=(0, 10))

        # Create a placeholder image
        self.preview_width = 380
        self.preview_height = 214
        placeholder = Image.new('RGB', (self.preview_width, self.preview_height), color=(26, 26, 26))
        self.preview_image = ctk.CTkImage(
            light_image=placeholder,
            dark_image=placeholder,
            size=(self.preview_width, self.preview_height)
        )

        self.preview_canvas = ctk.CTkLabel(
            preview_container,
            image=self.preview_image,
            text=""
        )
        self.preview_canvas.pack()

        # Preview controls
        preview_controls = ctk.CTkFrame(preview_frame, fg_color="transparent")
        preview_controls.pack(fill="x", padx=15, pady=(0, 15))

        self.preview_btn = ctk.CTkButton(
            preview_controls,
            text="Play Preview",
            command=self._toggle_preview,
            width=120
        )
        self.preview_btn.pack(side="left")

        self.preview_fps_label = ctk.CTkLabel(
            preview_controls,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.preview_fps_label.pack(side="right")

    def _create_parameter_sliders(self, parent):
        """Create the parameter adjustment sliders."""
        # Sliders frame
        sliders_frame = ctk.CTkFrame(parent)
        sliders_frame.pack(fill="x", pady=(0, 15))

        sliders_label = ctk.CTkLabel(
            sliders_frame,
            text="Audio Parameters",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        sliders_label.pack(anchor="w", padx=15, pady=(15, 10))

        # Create slider variables
        self.param_vars = {
            'bass': ctk.DoubleVar(value=1.0),
            'mids': ctk.DoubleVar(value=1.0),
            'highs': ctk.DoubleVar(value=1.0),
            'beat': ctk.DoubleVar(value=1.0),
            'speed': ctk.DoubleVar(value=1.0),
        }

        # Slider definitions: (label, var_key, min, max, default)
        slider_defs = [
            ("Bass Response", 'bass', 0.0, 2.0, 1.0),
            ("Mids Response", 'mids', 0.0, 2.0, 1.0),
            ("Highs Response", 'highs', 0.0, 2.0, 1.0),
            ("Beat Sensitivity", 'beat', 0.0, 2.0, 1.0),
            ("Animation Speed", 'speed', 0.1, 3.0, 1.0),
        ]

        self.param_value_labels = {}

        for label_text, var_key, min_val, max_val, default in slider_defs:
            self._create_slider_row(
                sliders_frame, label_text, var_key,
                min_val, max_val, default
            )

        # Envelope parameters section
        env_label = ctk.CTkLabel(
            sliders_frame,
            text="Envelope Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        env_label.pack(anchor="w", padx=15, pady=(15, 10))

        self.param_vars['attack'] = ctk.DoubleVar(value=5.0)
        self.param_vars['release'] = ctk.DoubleVar(value=150.0)

        self._create_slider_row(sliders_frame, "Attack (ms)", 'attack', 1.0, 50.0, 5.0)
        self._create_slider_row(sliders_frame, "Release (ms)", 'release', 20.0, 500.0, 150.0)

        # Preset buttons row
        preset_btn_row = ctk.CTkFrame(sliders_frame, fg_color="transparent")
        preset_btn_row.pack(fill="x", padx=15, pady=(15, 5))

        save_preset_btn = ctk.CTkButton(
            preset_btn_row,
            text="Save Preset",
            command=self._save_preset,
            width=100,
            height=28,
            fg_color="#4a4a4a",
            hover_color="#5a5a5a"
        )
        save_preset_btn.pack(side="left", padx=(0, 10))

        load_preset_btn = ctk.CTkButton(
            preset_btn_row,
            text="Load Preset",
            command=self._load_preset,
            width=100,
            height=28,
            fg_color="#4a4a4a",
            hover_color="#5a5a5a"
        )
        load_preset_btn.pack(side="left")

        # Reset button
        reset_btn = ctk.CTkButton(
            sliders_frame,
            text="Reset to Defaults",
            command=self._reset_parameters,
            width=140,
            height=28,
            fg_color="#4a4a4a",
            hover_color="#5a5a5a"
        )
        reset_btn.pack(anchor="w", padx=15, pady=(10, 15))

    def _create_slider_row(self, parent, label_text, var_key, min_val, max_val, default):
        """Create a single slider row with label and value display."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=15, pady=(0, 8))

        label = ctk.CTkLabel(row, text=label_text, width=120, anchor="w")
        label.pack(side="left")

        slider = ctk.CTkSlider(
            row,
            from_=min_val,
            to=max_val,
            variable=self.param_vars[var_key],
            command=lambda v, k=var_key: self._on_param_change(k, v),
            width=150
        )
        slider.pack(side="left", padx=(5, 5))

        # Value label
        if var_key in ['attack', 'release']:
            value_text = f"{default:.0f}"
        else:
            value_text = f"{default:.2f}"

        value_label = ctk.CTkLabel(row, text=value_text, width=50, anchor="e")
        value_label.pack(side="right")
        self.param_value_labels[var_key] = value_label

    def _on_param_change(self, key, value):
        """Handle parameter slider changes."""
        if key in ['attack', 'release']:
            self.param_value_labels[key].configure(text=f"{value:.0f}")
        else:
            self.param_value_labels[key].configure(text=f"{value:.2f}")

    def _reset_parameters(self):
        """Reset all parameters to default values."""
        defaults = {
            'bass': 1.0, 'mids': 1.0, 'highs': 1.0,
            'beat': 1.0, 'speed': 1.0,
            'attack': 5.0, 'release': 150.0
        }
        for key, val in defaults.items():
            self.param_vars[key].set(val)
            self._on_param_change(key, val)

    def _save_preset(self):
        """Save current parameter values to a preset file."""
        import json
        from tkinter import filedialog

        # Get current shader name for default filename
        shader_name = self.shader_combo.get()

        preset_dir = get_app_path() / "presets"
        preset_dir.mkdir(exist_ok=True)

        filepath = filedialog.asksaveasfilename(
            initialdir=str(preset_dir),
            initialfile=f"{shader_name}_preset.json",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filepath:
            preset_data = {
                'shader': shader_name,
                'zoom': self.zoom_var.get(),
                'parameters': {
                    key: var.get() for key, var in self.param_vars.items()
                }
            }
            with open(filepath, 'w') as f:
                json.dump(preset_data, f, indent=2)
            self.progress_label.configure(text=f"Preset saved: {Path(filepath).name}")

    def _load_preset(self):
        """Load parameter values from a preset file."""
        import json
        from tkinter import filedialog

        preset_dir = get_app_path() / "presets"
        preset_dir.mkdir(exist_ok=True)

        filepath = filedialog.askopenfilename(
            initialdir=str(preset_dir),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filepath:
            try:
                with open(filepath, 'r') as f:
                    preset_data = json.load(f)

                # Load shader if specified
                if 'shader' in preset_data:
                    shaders = self._get_shader_list()
                    if preset_data['shader'] in shaders:
                        self.shader_combo.set(preset_data['shader'])

                # Load zoom
                if 'zoom' in preset_data:
                    self.zoom_var.set(preset_data['zoom'])
                    self._on_zoom_change(preset_data['zoom'])

                # Load parameters
                if 'parameters' in preset_data:
                    for key, val in preset_data['parameters'].items():
                        if key in self.param_vars:
                            self.param_vars[key].set(val)
                            self._on_param_change(key, val)

                self.progress_label.configure(text=f"Preset loaded: {Path(filepath).name}")

            except Exception as e:
                self.progress_label.configure(text=f"Error loading preset: {e}")

    def _toggle_preview(self):
        """Start or stop the shader preview."""
        if self.preview_running:
            self._stop_preview()
        else:
            self._start_preview()

    def _start_preview(self):
        """Start the preview rendering loop."""
        if self.preview_running:
            return

        self.preview_running = True
        self.preview_stop_requested = False
        self.preview_btn.configure(text="Stop Preview")

        self.preview_thread = Thread(target=self._preview_worker, daemon=True)
        self.preview_thread.start()

    def _stop_preview(self):
        """Stop the preview rendering loop."""
        if not self.preview_running:
            return

        self.preview_stop_requested = True
        self.preview_btn.configure(text="Stopping...", state="disabled")

    def _preview_worker(self):
        """Background worker for preview rendering."""
        from .synthetic_audio import SyntheticAudioGenerator
        from .renderer import ShaderRenderer

        renderer = None
        try:
            # Initialize at lower resolution for performance
            preview_render_width = 480
            preview_render_height = 270

            renderer = ShaderRenderer(preview_render_width, preview_render_height)

            # Load the currently selected shader
            app_path = get_app_path()
            shader_name = self.shader_combo.get()
            shader_path = app_path / "shaders" / f"{shader_name}.glsl"
            renderer.load_shader(str(shader_path))

            # Create synthetic audio generator
            audio_gen = SyntheticAudioGenerator(fps=20)

            target_fps = 20
            frame_time = 1.0 / target_fps
            frame = 0

            while not self.preview_stop_requested:
                start = time.time()

                # Get current parameter values
                multipliers = {
                    'bass': self.param_vars['bass'].get(),
                    'mids': self.param_vars['mids'].get(),
                    'highs': self.param_vars['highs'].get(),
                    'beat': self.param_vars['beat'].get(),
                    'speed': self.param_vars['speed'].get(),
                }

                # Update envelope parameters
                audio_gen.set_envelope_params(
                    self.param_vars['attack'].get(),
                    self.param_vars['release'].get()
                )

                # Get synthetic audio data
                audio_data = audio_gen.get_frame_data(multipliers)

                # Render frame
                zoom = self.zoom_var.get()
                pixels = renderer.render_frame_rgb(
                    audio_data['time'], frame, audio_data, zoom, 1.0
                )

                # Convert to PIL Image and resize for display
                img = Image.fromarray(pixels)
                img = img.resize((self.preview_width, self.preview_height), Image.LANCZOS)

                # Update the preview (must be done on main thread)
                self.after(0, lambda i=img: self._update_preview_image(i))

                frame += 1

                # Maintain target FPS
                elapsed = time.time() - start
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Update FPS display every 10 frames
                if frame % 10 == 0:
                    actual_fps = 1.0 / max(elapsed, 0.001)
                    self.after(0, lambda f=actual_fps: self.preview_fps_label.configure(
                        text=f"{f:.0f} fps"
                    ))

        except Exception as e:
            print(f"Preview error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if renderer:
                try:
                    renderer.close()
                except:
                    pass

            self.preview_running = False
            self.after(0, lambda: self.preview_btn.configure(
                text="Play Preview", state="normal"
            ))
            self.after(0, lambda: self.preview_fps_label.configure(text=""))

    def _update_preview_image(self, img: Image.Image):
        """Update the preview canvas with a new image."""
        try:
            self.preview_image = ctk.CTkImage(
                light_image=img,
                dark_image=img,
                size=(self.preview_width, self.preview_height)
            )
            self.preview_canvas.configure(image=self.preview_image)
        except Exception as e:
            pass  # Ignore errors during shutdown

    def _setup_drag_drop(self):
        """Setup drag and drop for audio files."""
        # Note: tkinterdnd2 would be needed for full drag-drop support
        # For now, users use the Browse button
        pass

    def _browse_file(self):
        """Open file browser for audio selection."""
        filetypes = [
            ("Audio files", "*.mp3 *.wav *.flac *.ogg *.m4a"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.audio_path = path
            self.file_entry.configure(state="normal")
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, Path(path).name)
            self.file_entry.configure(state="disabled")

    def _browse_output(self):
        """Open folder browser for output selection."""
        path = filedialog.askdirectory(initialdir=str(self.output_dir))
        if path:
            self.output_dir = Path(path)
            self._update_output_display()

    def _update_output_display(self):
        """Update the output folder display."""
        self.output_entry.configure(state="normal")
        self.output_entry.delete(0, "end")
        # Show shortened path if too long
        display_path = str(self.output_dir)
        if len(display_path) > 40:
            display_path = "..." + display_path[-37:]
        self.output_entry.insert(0, display_path)
        self.output_entry.configure(state="disabled")

    def _on_preset_change(self, value: str):
        """Handle resolution preset change."""
        presets = {
            "1920x1080 (1080p)": (1920, 1080),
            "1280x720 (720p)": (1280, 720),
            "3840x2160 (4K)": (3840, 2160),
            "720x1280 (Vertical 720p)": (720, 1280),
            "1080x1920 (Vertical 1080p)": (1080, 1920),
        }

        if value in presets:
            w, h = presets[value]
            self.width_entry.delete(0, "end")
            self.width_entry.insert(0, str(w))
            self.height_entry.delete(0, "end")
            self.height_entry.insert(0, str(h))

    def _start_render(self):
        """Start the rendering process."""
        if self.rendering:
            # If already rendering, this acts as a stop button
            self._stop_render()
            return

        if not self.audio_path:
            self.progress_label.configure(text="Please select an audio file")
            return

        try:
            width = int(self.width_entry.get())
            height = int(self.height_entry.get())
        except ValueError:
            self.progress_label.configure(text="Invalid resolution")
            return

        self.rendering = True
        self.stop_requested = False
        self.render_btn.configure(text="Stop Render", fg_color="#8B0000", hover_color="#A52A2A")
        self.progress_bar.set(0)

        # Capture values before starting thread
        zoom = self.zoom_var.get()
        param_values = {key: var.get() for key, var in self.param_vars.items()}

        # Start render in background thread
        self.render_thread = Thread(
            target=self._render_worker,
            args=(width, height, zoom, param_values),
            daemon=True
        )
        self.render_thread.start()

    def _stop_render(self):
        """Request the render to stop."""
        if self.rendering:
            self.stop_requested = True
            self.render_btn.configure(state="disabled", text="Stopping...")
            self._update_progress("Stopping render...", self.progress_bar.get())

    def _render_worker(self, width: int, height: int, zoom: float = 1.0, param_values: dict = None):
        """Background worker for rendering."""
        import traceback
        renderer = None
        exporter = None
        was_stopped = False

        if param_values is None:
            param_values = {}

        try:
            from .audio import AudioAnalyzer
            from .renderer import ShaderRenderer as Renderer
            from .exporter import VideoExporter as Exporter

            self._update_progress("Analyzing audio...", 0)

            # Load audio with envelope parameters from sliders
            audio = AudioAnalyzer(self.audio_path, fps=60)
            # Configure envelope times if provided
            if 'attack' in param_values and 'release' in param_values:
                audio.envelope_attack_ms = param_values['attack']
                audio.envelope_release_ms = param_values['release']
            audio.load()

            # Setup output path (use user-selected directory)
            self.output_dir.mkdir(exist_ok=True)
            audio_name = Path(self.audio_path).stem
            shader_name = self.shader_combo.get()

            # Add shader name and preview suffix if not rendering full song
            if self.duration_var.get() == "preview":
                output_path = str(self.output_dir / f"{audio_name}_{shader_name}_preview.mp4")
            else:
                output_path = str(self.output_dir / f"{audio_name}_{shader_name}.mp4")

            # Setup shader
            app_path = get_app_path()
            shader_path = app_path / "shaders" / f"{self.shader_combo.get()}.glsl"

            self._update_progress("Initializing renderer...", 0.05)

            # Initialize renderer
            renderer = Renderer(width, height)
            renderer.load_shader(str(shader_path))

            # Initialize exporter
            exporter = Exporter(
                output_path=output_path,
                width=width,
                height=height,
                fps=60,
                audio_path=self.audio_path,
                codec="h264",
                bitrate="10M",
            )

            exporter.start()

            # Determine frame count based on duration setting
            total_frames = audio.frame_count
            if self.duration_var.get() == "preview":
                try:
                    preview_secs = float(self.preview_seconds.get())
                    preview_frames = int(preview_secs * 60)  # 60 fps
                    total_frames = min(preview_frames, total_frames)
                except ValueError:
                    pass  # Use full length if invalid input

            self._update_progress(f"Rendering {total_frames} frames...", 0.1)

            # Fade out duration (1 second = 60 frames at 60fps)
            fade_out_frames = 60

            # Render loop with timing
            import time
            start_time = time.time()
            last_update = start_time

            for frame in range(total_frames):
                # Check if stop was requested
                if self.stop_requested:
                    was_stopped = True
                    break

                audio_data = audio.get_frame_data(frame)
                frame_time = audio_data['time']

                # Apply parameter multipliers to audio data
                if 'bass' in param_values:
                    audio_data['bass'] = min(1.0, audio_data['bass'] * param_values['bass'])
                if 'mids' in param_values:
                    audio_data['mids'] = min(1.0, audio_data['mids'] * param_values['mids'])
                if 'highs' in param_values:
                    audio_data['highs'] = min(1.0, audio_data['highs'] * param_values['highs'])
                if 'beat' in param_values:
                    audio_data['beat'] = min(1.0, audio_data['beat'] * param_values['beat'])
                if 'speed' in param_values:
                    audio_data['audio_time'] = audio_data['audio_time'] * param_values['speed']

                # Calculate fade (1.0 = full brightness, 0.0 = black)
                frames_remaining = total_frames - frame - 1
                if frames_remaining < fade_out_frames:
                    fade = frames_remaining / fade_out_frames
                else:
                    fade = 1.0

                pixels = renderer.render_frame_rgb(frame_time, frame, audio_data, zoom, fade)
                exporter.write_frame(pixels)

                # Update progress every 0.1 seconds (not frame count)
                now = time.time()
                if now - last_update >= 0.1:
                    last_update = now
                    elapsed = now - start_time
                    fps = frame / elapsed if elapsed > 0 else 0
                    progress = frame / total_frames
                    pct = progress * 100
                    self._update_progress(
                        f"Frame {frame}/{total_frames} ({pct:.0f}%) - {fps:.0f} fps",
                        progress
                    )

            if was_stopped:
                self._update_progress("Render stopped by user", 0)
                # Clean up without saving partial video
                exporter.running = False  # Stop the encoding thread
                if exporter.process:
                    exporter.process.stdin.close()
                    exporter.process.terminate()
                exporter = None
                renderer.close()
                renderer = None
            else:
                self._update_progress("Finalizing video...", 0.95)
                success = exporter.finish()
                exporter = None  # Mark as cleaned up
                renderer.close()
                renderer = None  # Mark as cleaned up

                if success:
                    self._update_progress(f"Done! Saved to outputs/", 1.0)
                else:
                    self._update_progress("Encoding failed", 0)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"Render error: {e}")
            traceback.print_exc()
            self._update_progress(error_msg, 0)

        finally:
            # Always clean up resources, even on error
            if renderer is not None:
                try:
                    renderer.close()
                except Exception as cleanup_err:
                    print(f"Error during renderer cleanup: {cleanup_err}")

            if exporter is not None:
                try:
                    exporter.finish()
                except Exception as cleanup_err:
                    print(f"Error during exporter cleanup: {cleanup_err}")

            self.rendering = False
            self.stop_requested = False
            self.after(0, lambda: self.render_btn.configure(
                state="normal", text="Start Render",
                fg_color=("#3B8ED0", "#1F6AA5"),  # Default CTkButton blue
                hover_color=("#36719F", "#144870")
            ))

    def _update_progress(self, text: str, value: float):
        """Update progress bar and label from worker thread."""
        self.after(0, lambda: self.progress_label.configure(text=text))
        self.after(0, lambda: self.progress_bar.set(value))


def main():
    """Launch the GUI application."""
    app = VisualizerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
