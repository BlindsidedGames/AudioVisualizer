"""Video exporter using FFmpeg for encoding frames to video."""

import subprocess
import shutil
import numpy as np
from pathlib import Path


def _find_ffmpeg() -> str:
    """Find ffmpeg executable, checking common locations."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    winget_path = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    if winget_path.exists():
        for ffmpeg_dir in winget_path.glob("Gyan.FFmpeg*"):
            for bin_dir in ffmpeg_dir.rglob("bin"):
                ffmpeg_exe = bin_dir / "ffmpeg.exe"
                if ffmpeg_exe.exists():
                    return str(ffmpeg_exe)

    return "ffmpeg"


class VideoExporter:
    """Exports frames to video using FFmpeg with optional hardware encoding."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: int = 60,
        audio_path: str = None,
        codec: str = "h264",
        use_nvenc: bool = True,
        bitrate: str = "10M",
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.audio_path = audio_path
        self.codec = codec
        self.use_nvenc = use_nvenc
        self.bitrate = bitrate
        self.process = None

    def _get_encoder(self) -> str:
        """Get the appropriate encoder based on settings."""
        if self.use_nvenc:
            if self.codec == "h264":
                return "h264_nvenc"
            elif self.codec == "hevc":
                return "hevc_nvenc"
        # Fallback to software encoding
        if self.codec == "h264":
            return "libx264"
        elif self.codec == "hevc":
            return "libx265"
        return "libx264"

    def start(self):
        """Start the FFmpeg process for frame input."""
        encoder = self._get_encoder()
        ffmpeg_path = _find_ffmpeg()

        cmd = [
            ffmpeg_path,
            "-y",  # Overwrite output
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",  # Read from stdin
        ]

        # Add audio if provided
        if self.audio_path:
            cmd.extend(["-i", self.audio_path])

        # Video encoding settings
        cmd.extend([
            "-c:v", encoder,
            "-b:v", self.bitrate,
            "-pix_fmt", "yuv420p",
        ])

        # Encoder-specific settings
        if "nvenc" in encoder:
            cmd.extend(["-preset", "p4", "-tune", "hq"])
        elif encoder == "libx264":
            cmd.extend(["-preset", "medium", "-crf", "18"])
        elif encoder == "libx265":
            cmd.extend(["-preset", "medium", "-crf", "20"])

        # Audio settings
        if self.audio_path:
            cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])

        cmd.append(self.output_path)

        print(f"Starting FFmpeg with encoder: {encoder}")
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write_frame(self, frame: np.ndarray):
        """Write a frame to the video."""
        if self.process is None:
            raise RuntimeError("Exporter not started. Call start() first.")
        self.process.stdin.write(frame.tobytes())

    def finish(self):
        """Finish encoding and close the file."""
        if self.process:
            self.process.stdin.close()

            # Use communicate() to avoid deadlock from full stderr pipe
            _, stderr = self.process.communicate(timeout=60)

            # Check for errors
            if self.process.returncode != 0:
                stderr_str = stderr.decode() if stderr else ""
                if "nvenc" in stderr_str.lower() and "failed" in stderr_str.lower():
                    print("NVENC encoding failed. Retrying with software encoder...")
                    return False
                print(f"FFmpeg error: {stderr_str}")
                return False

            print(f"Video saved to: {self.output_path}")
            return True
        return False


class SoftwareVideoExporter(VideoExporter):
    """Video exporter that always uses software encoding (no GPU)."""

    def __init__(self, *args, **kwargs):
        kwargs['use_nvenc'] = False
        super().__init__(*args, **kwargs)
