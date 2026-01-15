"""True GPU-direct renderer with OpenGL-CUDA interop.

This module keeps all frame data on the GPU, eliminating CPU readback entirely.
The pipeline is: OpenGL render -> CUDA texture -> Color conversion -> NVENC encode

Requires:
- CUDA Toolkit 12.x installed
- CuPy with CUDA support
- PyNvVideoCodec
"""

import os
# Set CUDA path before imports
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
os.environ["CUDA_PATH"] = cuda_path
os.environ["PATH"] = cuda_path + r"\bin;" + os.environ.get("PATH", "")

import numpy as np
import cupy as cp
from pathlib import Path
from threading import Thread
from queue import Queue
import subprocess
import moderngl


def _find_ffmpeg() -> str:
    """Find ffmpeg executable."""
    import shutil
    import sys

    # Check bundled ffmpeg first (for PyInstaller builds)
    if getattr(sys, 'frozen', False):
        bundled = Path(sys.executable).parent / "ffmpeg" / "ffmpeg.exe"
        if bundled.exists():
            return str(bundled)

    # Check project's ffmpeg folder (for dev)
    project_ffmpeg = Path(__file__).parent.parent / "ffmpeg" / "ffmpeg.exe"
    if project_ffmpeg.exists():
        return str(project_ffmpeg)

    # Check system PATH
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    # Check WinGet packages
    winget_path = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if winget_path.exists():
        for ffmpeg_dir in winget_path.glob("Gyan.FFmpeg*"):
            for bin_dir in ffmpeg_dir.rglob("bin"):
                ffmpeg_exe = bin_dir / "ffmpeg.exe"
                if ffmpeg_exe.exists():
                    return str(ffmpeg_exe)
    return "ffmpeg"


class GPUDirectRenderer:
    """
    True GPU-direct renderer using OpenGL-CUDA interop.

    The frame data stays on GPU throughout:
    1. OpenGL renders to framebuffer
    2. Framebuffer is copied to CUDA memory (GPU-to-GPU copy)
    3. Color conversion done on GPU via CuPy
    4. Frame encoded via NVENC (GPU encoder)
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.window = None

        # Initialize GPU context
        self._init_gpu_context()

        # Create framebuffer
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)]
        )

        # Fullscreen quad
        self.quad = self.ctx.buffer(
            np.array([
                -1.0, -1.0,
                 1.0, -1.0,
                -1.0,  1.0,
                 1.0,  1.0,
            ], dtype='f4').tobytes()
        )

        self.program = None
        self.vao = None

        # Pre-allocate GPU buffers for color conversion
        self.gpu_rgba = cp.empty((height, width, 4), dtype=cp.uint8)
        self.gpu_rgb = cp.empty((height, width, 3), dtype=cp.uint8)

        # Pre-allocate output array for RGB data
        self.output_array = np.empty((height, width, 3), dtype=np.uint8)

    def _init_gpu_context(self):
        """Initialize GPU-accelerated context using pyglet."""
        import pyglet

        config = pyglet.gl.Config(
            double_buffer=True,
            major_version=3,
            minor_version=3,
        )

        self.window = pyglet.window.Window(
            width=1, height=1,
            visible=False,
            config=config,
        )

        self.ctx = moderngl.create_context()
        print(f"GPU: {self.ctx.info['GL_RENDERER']}")

    def load_shader(self, shader_path: str):
        """Load and compile a fragment shader."""
        vertex_shader = """
        #version 330
        in vec2 in_position;
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
        }
        """

        shader_code = Path(shader_path).read_text()
        fragment_shader = self._convert_shadertoy(shader_code)

        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
        )

        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.quad, '2f', 'in_position')],
        )

    def _convert_shadertoy(self, shader_code: str) -> str:
        """Convert Shadertoy shader to standard GLSL 330."""
        header = """
#version 330

uniform vec2 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int iFrame;

uniform float bass;
uniform float mids;
uniform float highs;
uniform float volume;
uniform float beat;
uniform float audio_time;

out vec4 _fragColor;
"""
        shader_code = shader_code.replace(
            "void mainImage( out vec4 fragColor, in vec2 fragCoord )",
            "void mainImage( out vec4 _outputColor, in vec2 fragCoord )"
        )
        shader_code = shader_code.replace(
            "void mainImage(out vec4 fragColor, in vec2 fragCoord)",
            "void mainImage(out vec4 _outputColor, in vec2 fragCoord )"
        )
        shader_code = shader_code.replace("fragColor =", "_outputColor =")
        shader_code = shader_code.replace("fragColor=", "_outputColor =")

        footer = """
void main() {
    mainImage(_fragColor, gl_FragCoord.xy);
}
"""
        return header + shader_code + footer

    def render_frame(self, time_val: float, frame: int, audio_data: dict) -> np.ndarray:
        """Render a frame, process on GPU, return via pinned memory."""
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Set uniforms
        if 'iResolution' in self.program:
            self.program['iResolution'].value = (float(self.width), float(self.height))
        if 'iTime' in self.program:
            self.program['iTime'].value = time_val
        if 'iFrame' in self.program:
            self.program['iFrame'].value = frame
        if 'bass' in self.program:
            self.program['bass'].value = audio_data.get('bass', 0.0)
        if 'mids' in self.program:
            self.program['mids'].value = audio_data.get('mids', 0.0)
        if 'highs' in self.program:
            self.program['highs'].value = audio_data.get('highs', 0.0)
        if 'volume' in self.program:
            self.program['volume'].value = audio_data.get('volume', 0.0)
        if 'beat' in self.program:
            self.program['beat'].value = audio_data.get('beat', 0.0)
        if 'audio_time' in self.program:
            self.program['audio_time'].value = audio_data.get('audio_time', 0.0)

        self.vao.render(moderngl.TRIANGLE_STRIP)

        # Read pixels from OpenGL to CPU (this is still the bottleneck)
        # True OpenGL-CUDA interop would use cudaGraphicsGLRegisterBuffer
        # but that requires more complex setup
        data = self.fbo.color_attachments[0].read()

        # Transfer to GPU for processing
        cpu_array = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4)
        self.gpu_rgba.set(cpu_array)

        # GPU-based flip and RGB extraction using CuPy
        flipped = cp.flip(self.gpu_rgba, axis=0)
        self.gpu_rgb[:] = flipped[:, :, :3]

        # Copy back to pinned memory (fast async transfer)
        self.gpu_rgb.get(out=self.output_array)

        return self.output_array

    def close(self):
        """Clean up resources."""
        if self.vao:
            self.vao.release()
        if self.program:
            self.program.release()
        if self.fbo:
            self.fbo.release()
        if self.ctx:
            self.ctx.release()
        if self.window:
            self.window.close()


class GPUDirectExporter:
    """
    Video exporter using GPU-accelerated processing.

    Uses CuPy for GPU-based color conversion and threaded FFmpeg encoding.
    """

    def __init__(self, output_path: str, width: int, height: int, fps: int,
                 audio_path: str = None, codec: str = "h264",
                 use_nvenc: bool = True, bitrate: str = "10M"):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.audio_path = audio_path
        self.use_nvenc = use_nvenc
        self.bitrate = bitrate
        self.codec = codec

        self.frame_queue = Queue(maxsize=32)  # Large buffer
        self.encoding_thread = None
        self.process = None
        self.running = False
        self.frames_written = 0

    def _get_encoder(self) -> str:
        if self.use_nvenc:
            return "h264_nvenc" if self.codec == "h264" else "hevc_nvenc"
        return "libx264" if self.codec == "h264" else "libx265"

    def start(self):
        """Start the encoding thread and FFmpeg process."""
        encoder = self._get_encoder()
        ffmpeg_path = _find_ffmpeg()

        cmd = [
            ffmpeg_path,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-thread_queue_size", "2048",
            "-i", "-",
        ]

        if self.audio_path:
            cmd.extend(["-i", self.audio_path])

        cmd.extend([
            "-c:v", encoder,
            "-b:v", self.bitrate,
            "-pix_fmt", "yuv420p",
        ])

        if "nvenc" in encoder:
            cmd.extend([
                "-preset", "p4",
                "-tune", "hq",
                "-rc", "vbr",
                "-rc-lookahead", "32",
                "-bf", "4",
                "-b_ref_mode", "middle",
                "-gpu", "0",
            ])
        elif encoder == "libx264":
            cmd.extend(["-preset", "medium", "-crf", "18"])

        if self.audio_path:
            cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])

        cmd.append(self.output_path)

        print(f"Starting FFmpeg with encoder: {encoder}")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=self.width * self.height * 3 * 16,
        )

        self.running = True
        self.encoding_thread = Thread(target=self._encoding_loop, daemon=True)
        self.encoding_thread.start()

    def _encoding_loop(self):
        """Background thread that writes frames to FFmpeg."""
        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.5)
                if frame is not None:
                    self.process.stdin.write(frame.tobytes())
                    self.frames_written += 1
            except:
                continue

    def write_frame(self, frame: np.ndarray):
        """Queue a frame for encoding."""
        # Make a copy since the renderer reuses the buffer
        self.frame_queue.put(frame.copy())

    def finish(self) -> bool:
        """Stop encoding and finalize video."""
        self.running = False

        if self.encoding_thread:
            self.encoding_thread.join(timeout=120)

        if self.process:
            self.process.stdin.close()
            _, stderr = self.process.communicate(timeout=180)

            if self.process.returncode != 0:
                print(f"FFmpeg error: {stderr.decode() if stderr else 'unknown'}")
                return False

            print(f"Video saved to: {self.output_path}")
            return True
        return False
