"""GUI interface for the visualizer tool using CustomTkinter."""

import customtkinter as ctk
from tkinter import filedialog
from pathlib import Path
from threading import Thread
import sys
import os


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
        self.geometry("500x950")
        self.resizable(False, False)

        # Set appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # State
        self.audio_path = None
        self.output_dir = get_output_path()
        self.rendering = False
        self.render_thread = None

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
        """Create all UI widgets."""
        # Main container with padding
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=20, pady=20)

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

        # CUDA install hint - only show when running from source (not bundled exe)
        # Bundled exe users should check README for GPU acceleration instructions
        if not self.capabilities["cuda"] and not self.capabilities["bundled"]:
            hint_frame = ctk.CTkFrame(container, fg_color="#2b2b2b")
            hint_frame.pack(fill="x", pady=(0, 10))

            hint_row = ctk.CTkFrame(hint_frame, fg_color="transparent")
            hint_row.pack(fill="x", padx=10, pady=8)

            hint_label = ctk.CTkLabel(
                hint_row,
                text="Want 5x faster rendering?",
                font=ctk.CTkFont(size=11)
            )
            hint_label.pack(side="left")

            link_btn = ctk.CTkButton(
                hint_row,
                text="Enable GPU Mode",
                command=self._open_cuda_link,
                width=120,
                height=24,
                fg_color="#4a4a4a",
                hover_color="#5a5a5a",
                font=ctk.CTkFont(size=11)
            )
            link_btn.pack(side="right")

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

        # Find available shaders
        shader_dir = get_app_path() / "shaders"
        shaders = [f.stem for f in shader_dir.glob("*.glsl")] if shader_dir.exists() else ["universe_within"]

        self.shader_combo = ctk.CTkComboBox(
            shader_frame,
            values=shaders,
            width=200
        )
        self.shader_combo.set(shaders[0] if shaders else "universe_within")
        self.shader_combo.pack(anchor="w", padx=15, pady=(0, 15))

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

        # Render mode section (only show if GPU is available)
        if self.capabilities["cupy"] or self.capabilities["cuda"]:
            mode_frame = ctk.CTkFrame(container)
            mode_frame.pack(fill="x", pady=(0, 15))

            mode_label = ctk.CTkLabel(
                mode_frame,
                text="Render Mode",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            mode_label.pack(anchor="w", padx=15, pady=(15, 5))

            mode_row = ctk.CTkFrame(mode_frame, fg_color="transparent")
            mode_row.pack(fill="x", padx=15, pady=(0, 15))

            self.render_mode = ctk.StringVar(value="gpu")

            gpu_radio = ctk.CTkRadioButton(
                mode_row,
                text="GPU Accelerated (faster)",
                variable=self.render_mode,
                value="gpu"
            )
            gpu_radio.pack(side="left", padx=(0, 20))

            std_radio = ctk.CTkRadioButton(
                mode_row,
                text="Standard (compatibility)",
                variable=self.render_mode,
                value="standard"
            )
            std_radio.pack(side="left")
        else:
            self.render_mode = ctk.StringVar(value="standard")

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

    def _get_status_text(self) -> str:
        """Get status text based on capabilities."""
        parts = []
        if self.capabilities["cupy"]:
            parts.append("GPU Accelerated (CuPy)")
        elif self.capabilities["cuda"]:
            parts.append("GPU Accelerated (CUDA)")
        else:
            parts.append("Standard Mode")

        if self.capabilities["nvenc"]:
            parts.append("NVENC")

        return " | ".join(parts)

    def _setup_drag_drop(self):
        """Setup drag and drop for audio files."""
        # Note: tkinterdnd2 would be needed for full drag-drop support
        # For now, users use the Browse button
        pass

    def _open_cuda_link(self):
        """Show GPU acceleration instructions."""
        from tkinter import messagebox
        msg = """To enable 5x faster GPU rendering:

1. Install CUDA Toolkit 12.x from:
   nvidia.com/cuda-downloads

2. Open a command prompt and run:
   pip install cupy-cuda12x

3. Restart the Audio Visualizer

Requires an NVIDIA GPU with CUDA support."""
        messagebox.showinfo("Enable GPU Acceleration", msg)

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
        self.render_btn.configure(state="disabled", text="Rendering...")
        self.progress_bar.set(0)

        # Start render in background thread
        self.render_thread = Thread(
            target=self._render_worker,
            args=(width, height),
            daemon=True
        )
        self.render_thread.start()

    def _render_worker(self, width: int, height: int):
        """Background worker for rendering."""
        import traceback
        renderer = None
        exporter = None

        try:
            from .audio import AudioAnalyzer

            # Check user's render mode selection
            use_gpu = self.render_mode.get() == "gpu"

            # Choose renderer based on capabilities and user selection
            if use_gpu and self.capabilities["cupy"]:
                from .renderer_gpudirect import GPUDirectRenderer as Renderer
                from .renderer_gpudirect import GPUDirectExporter as Exporter
                renderer_mode = "gpudirect"
            elif use_gpu and self.capabilities["cuda"]:
                from .renderer_cuda import CudaShaderRenderer as Renderer
                from .renderer_cuda import HybridExporter as Exporter
                renderer_mode = "cuda"
            else:
                from .renderer import ShaderRenderer as Renderer
                from .exporter import VideoExporter as Exporter
                renderer_mode = "standard"

            self._update_progress("Analyzing audio...", 0)

            # Load audio
            audio = AudioAnalyzer(self.audio_path, fps=60)
            audio.load()

            # Setup output path (use user-selected directory)
            self.output_dir.mkdir(exist_ok=True)
            audio_name = Path(self.audio_path).stem

            # Add preview suffix if not rendering full song
            if self.duration_var.get() == "preview":
                output_path = str(self.output_dir / f"{audio_name}_preview.mp4")
            else:
                output_path = str(self.output_dir / f"{audio_name}_visualized.mp4")

            # Setup shader
            app_path = get_app_path()
            shader_path = app_path / "shaders" / f"{self.shader_combo.get()}.glsl"

            self._update_progress("Initializing renderer...", 0.05)

            # Initialize renderer
            renderer = Renderer(width, height)
            renderer.load_shader(str(shader_path))

            # Initialize exporter
            if renderer_mode in ("gpudirect", "cuda"):
                exporter = Exporter(
                    output_path=output_path,
                    width=width,
                    height=height,
                    fps=60,
                    audio_path=self.audio_path,
                    codec="h264",
                    use_nvenc=self.capabilities["nvenc"],
                    bitrate="10M",
                )
            else:
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

            # Render loop with timing
            import time
            start_time = time.time()
            last_update = start_time

            for frame in range(total_frames):
                audio_data = audio.get_frame_data(frame)
                frame_time = audio_data['time']

                if renderer_mode in ("gpudirect", "cuda"):
                    pixels = renderer.render_frame(frame_time, frame, audio_data)
                else:
                    pixels = renderer.render_frame_rgb(frame_time, frame, audio_data)

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
            self.after(0, lambda: self.render_btn.configure(
                state="normal", text="Start Render"
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
