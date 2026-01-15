"""GPU-accelerated renderer with NVIDIA hardware encoding.

Uses PyTorch for CUDA operations and PyNvVideoCodec for hardware encoding.
"""

import torch
import numpy as np
import os
from pathlib import Path
from threading import Thread
from queue import Queue
import subprocess
import moderngl


def _find_ffmpeg() -> str:
    """Find ffmpeg executable, checking common locations."""
    # Check if in PATH
    import shutil
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    # Check winget installation location
    winget_path = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if winget_path.exists():
        for ffmpeg_dir in winget_path.glob("Gyan.FFmpeg*"):
            for bin_dir in ffmpeg_dir.rglob("bin"):
                ffmpeg_exe = bin_dir / "ffmpeg.exe"
                if ffmpeg_exe.exists():
                    return str(ffmpeg_exe)

    # Fallback to just "ffmpeg" and hope it's in PATH
    return "ffmpeg"


class CudaShaderRenderer:
    """
    Shader renderer with PyTorch-based GPU memory handling.

    Uses OpenGL for rendering and PyTorch for efficient GPU-to-CPU transfers.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.window = None

        # Initialize GPU context
        self._init_gpu_context()

        # Create framebuffer for offscreen rendering
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)]
        )

        # Fullscreen quad vertices
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

        # Pre-allocate PyTorch tensors for faster transfers
        self.device = torch.device('cuda:0')
        self.output_tensor = torch.empty((height, width, 4), dtype=torch.uint8, device='cpu', pin_memory=True)

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

out vec4 _fragColor;
"""
        shader_code = shader_code.replace(
            "void mainImage( out vec4 fragColor, in vec2 fragCoord )",
            "void mainImage( out vec4 _outputColor, in vec2 fragCoord )"
        )
        shader_code = shader_code.replace(
            "void mainImage(out vec4 fragColor, in vec2 fragCoord)",
            "void mainImage(out vec4 _outputColor, in vec2 fragCoord)"
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
        """Render a frame and return as numpy RGB array."""
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

        self.vao.render(moderngl.TRIANGLE_STRIP)

        # Read pixels into pinned memory tensor (faster transfer)
        data = self.fbo.color_attachments[0].read()
        np.copyto(self.output_tensor.numpy(), np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4))

        # Flip and extract RGB
        img = np.flip(self.output_tensor.numpy(), axis=0)
        return img[:, :, :3].copy()

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


class NvencExporter:
    """
    Video exporter using NVIDIA hardware encoder via PyNvVideoCodec.

    Uses threaded encoding with GPU-optimized memory transfers.
    """

    def __init__(self, output_path: str, width: int, height: int, fps: int,
                 audio_path: str = None, codec: str = "h264", bitrate: str = "10M"):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.audio_path = audio_path
        self.bitrate = bitrate
        self.codec = codec

        self.frame_queue = Queue(maxsize=16)  # Larger buffer for GPU encoding
        self.encoding_thread = None
        self.encoder = None
        self.running = False
        self.frames_written = 0
        self.encoded_data = bytearray()

        # Temporary file for raw video before muxing with audio
        self.temp_video = output_path + ".h264"

    def start(self):
        """Start the encoding thread and NVENC encoder."""
        import PyNvVideoCodec

        # Create NVENC encoder
        # fmt options: NV12, YV12, IYUV, YUV444, YUV420_10bit, YUV444_10bit, ARGB, ARGB10, AYUV, ABGR, ABGR10, RGB, BGR
        # We'll use NV12 which is the most efficient for video encoding

        encoder_opts = {
            'preset': 'P4',  # Quality preset (P1=fastest, P7=highest quality)
            'tuninginfo': 'high_quality',
            'rc': 'vbr',  # Variable bitrate
            'bitrate': int(self.bitrate.replace('M', '')) * 1_000_000,
            'maxbitrate': int(self.bitrate.replace('M', '')) * 1_500_000,
            'fps': self.fps,
        }

        if self.codec == "hevc":
            encoder_opts['codec'] = 'hevc'
        else:
            encoder_opts['codec'] = 'h264'

        print(f"Starting NVENC encoder: {self.codec}")

        # Create encoder with CPU input buffer (we'll convert RGB->NV12 ourselves)
        self.encoder = PyNvVideoCodec.CreateEncoder(
            self.width, self.height, 'NV12', True, **encoder_opts
        )

        self.running = True
        self.encoding_thread = Thread(target=self._encoding_loop, daemon=True)
        self.encoding_thread.start()

    def _rgb_to_nv12(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to NV12 format for NVENC."""
        # RGB to YUV BT.709
        r = rgb[:, :, 0].astype(np.float32)
        g = rgb[:, :, 1].astype(np.float32)
        b = rgb[:, :, 2].astype(np.float32)

        y = (0.2126 * r + 0.7152 * g + 0.0722 * b).astype(np.uint8)
        u = (128 - 0.1146 * r - 0.3854 * g + 0.5000 * b).astype(np.uint8)
        v = (128 + 0.5000 * r - 0.4542 * g - 0.0458 * b).astype(np.uint8)

        # Subsample U and V (4:2:0)
        u_sub = u[0::2, 0::2]
        v_sub = v[0::2, 0::2]

        # Create NV12 frame (Y plane followed by interleaved UV)
        nv12 = np.empty((self.height * 3 // 2, self.width), dtype=np.uint8)
        nv12[:self.height] = y
        uv = np.empty((self.height // 2, self.width), dtype=np.uint8)
        uv[:, 0::2] = u_sub
        uv[:, 1::2] = v_sub
        nv12[self.height:] = uv

        return nv12

    def _encoding_loop(self):
        """Background thread that encodes frames."""
        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.5)
                if frame is not None:
                    # Convert RGB to NV12
                    nv12 = self._rgb_to_nv12(frame)
                    # Encode
                    encoded = self.encoder.Encode(nv12)
                    if encoded:
                        self.encoded_data.extend(encoded)
                    self.frames_written += 1
            except:
                continue

    def write_frame(self, frame: np.ndarray):
        """Queue a frame for encoding."""
        self.frame_queue.put(frame)

    def finish(self) -> bool:
        """Stop encoding and finalize video."""
        self.running = False

        # Wait for queue to drain
        if self.encoding_thread:
            self.encoding_thread.join(timeout=60)

        # Flush encoder
        if self.encoder:
            final = self.encoder.EndEncode()
            if final:
                self.encoded_data.extend(final)

        # Write raw bitstream to temp file
        with open(self.temp_video, 'wb') as f:
            f.write(self.encoded_data)

        print(f"Encoded {self.frames_written} frames")

        # Mux with audio using FFmpeg
        return self._mux_with_audio()

    def _mux_with_audio(self) -> bool:
        """Mux encoded video with audio using FFmpeg."""
        ffmpeg_path = _find_ffmpeg()
        cmd = [
            ffmpeg_path, "-y",
            "-r", str(self.fps),
            "-i", self.temp_video,
        ]

        if self.audio_path:
            cmd.extend(["-i", self.audio_path])

        cmd.extend([
            "-c:v", "copy",  # Copy video stream (already encoded)
        ])

        if self.audio_path:
            cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])

        cmd.append(self.output_path)

        print("Muxing video with audio...")
        result = subprocess.run(cmd, capture_output=True)

        # Clean up temp file
        try:
            Path(self.temp_video).unlink()
        except:
            pass

        if result.returncode != 0:
            print(f"FFmpeg mux error: {result.stderr.decode()}")
            return False

        print(f"Video saved to: {self.output_path}")
        return True


class HybridExporter:
    """
    Hybrid exporter that uses NVENC through FFmpeg with optimized memory handling.

    This approach keeps the FFmpeg-based encoding (which is well-tested) but
    uses optimized memory transfers with pinned memory and better threading.
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

        # Larger queue for better pipelining
        self.frame_queue = Queue(maxsize=16)
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
            "-thread_queue_size", "1024",
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
            bufsize=self.width * self.height * 3 * 8,  # Buffer 8 frames
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
        """Queue a frame for encoding (non-blocking if queue not full)."""
        self.frame_queue.put(frame)

    def finish(self) -> bool:
        """Stop encoding and finalize video."""
        self.running = False

        # Wait for queue to drain
        if self.encoding_thread:
            self.encoding_thread.join(timeout=60)

        if self.process:
            self.process.stdin.close()
            _, stderr = self.process.communicate(timeout=120)

            if self.process.returncode != 0:
                print(f"FFmpeg error: {stderr.decode() if stderr else 'unknown'}")
                return False

            print(f"Video saved to: {self.output_path}")
            return True
        return False
