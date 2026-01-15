# Audio Visualizer

GPU-accelerated music visualizer that renders audio-reactive visuals to video at any resolution.

![Audio Visualizer GUI](https://img.shields.io/badge/Platform-Windows-blue) ![Python](https://img.shields.io/badge/Python-3.10--3.12-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **Audio-reactive visuals** - Shader-based visualizations that respond to bass, mids, highs, and beats
- **Any resolution** - Render at 1080p, 4K, vertical for social media, or custom dimensions
- **GPU acceleration** - Optional CUDA support for 5x faster rendering
- **Hardware encoding** - NVENC support for fast video export
- **Multiple presets** - Different visual styles and speed variants

## Quick Start

### 1. Install

Double-click `install.bat` or run it from command prompt.

The installer will:
- Find a compatible Python version (3.10-3.12)
- Create a virtual environment
- Install all dependencies
- Optionally install GPU acceleration (if CUDA is available)

### 2. Run

Double-click `run.bat` to launch the GUI.

## Requirements

### Required
- **Python 3.10, 3.11, or 3.12** - [Download Python 3.12](https://www.python.org/downloads/release/python-3129/)
- **FFmpeg** - Required for video encoding
  - Install via winget: `winget install ffmpeg`
  - Or download from [gyan.dev/ffmpeg](https://www.gyan.dev/ffmpeg/builds/) and add to PATH

### Optional (for GPU acceleration)
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit 12.x** - [Download CUDA 12.6](https://developer.nvidia.com/cuda-12-6-0-download-archive)

## Usage

1. **Select an audio file** - Supports MP3, WAV, FLAC, and other common formats
2. **Choose resolution** - Pick a preset or enter custom dimensions
3. **Select a visual preset** - Different styles and speeds available
4. **Choose output folder** - Where the video will be saved
5. **Click Start Render** - Progress bar shows rendering status

## Visual Presets

| Preset | Description |
|--------|-------------|
| `universe_within` | Default speed (75%) - Cosmic fractal visualization |
| `universe_within_medium` | Medium speed (50%) - Slower, more dramatic |
| `universe_within_slow` | Slow speed (25%) - Very slow, hypnotic |

## Rendering Modes

### Standard Mode (CPU)
- Works on any system
- No additional setup required
- Suitable for most uses

### GPU Accelerated Mode
- Requires NVIDIA GPU + CUDA Toolkit 12.x + CuPy
- ~5x faster rendering
- The installer will set this up automatically if CUDA is detected

To enable GPU mode after installation:
1. Install [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-12-6-0-download-archive)
2. Run: `venv\Scripts\pip.exe install cupy-cuda12x`
3. Restart the visualizer

## Troubleshooting

### "No compatible Python found"
Install Python 3.12 from [python.org](https://www.python.org/downloads/release/python-3129/). Make sure to check "Add Python to PATH" during installation.

### "FFmpeg not found"
Install FFmpeg via `winget install ffmpeg` or download from [gyan.dev/ffmpeg](https://www.gyan.dev/ffmpeg/builds/) and add the `bin` folder to your system PATH.

### GPU mode not working
- Ensure you have an NVIDIA GPU
- Install CUDA Toolkit 12.x (not 13.x)
- Run `venv\Scripts\pip.exe install cupy-cuda12x`
- Restart the visualizer

### Rendering is slow
- GPU mode is ~5x faster than Standard mode
- Lower resolutions render faster
- Close other GPU-intensive applications

## Command Line Usage

For advanced users, a CLI is also available:

```bash
venv\Scripts\python.exe main.py audio.mp3 --preset universe_within --resolution 1920x1080 --output video.mp4
```

Options:
- `--preset` - Visual preset name
- `--resolution` - Output resolution (WIDTHxHEIGHT)
- `--output` - Output file path
- `--fps` - Frame rate (default: 60)

## Project Structure

```
AudioVisualizer/
├── install.bat          # Installation script
├── run.bat              # GUI launcher
├── requirements.txt     # Python dependencies
├── run_gui.py           # GUI entry point
├── main.py              # CLI entry point
├── shaders/             # GLSL shader files
│   ├── universe_within.glsl
│   ├── universe_within_medium.glsl
│   └── universe_within_slow.glsl
└── src/
    ├── gui.py           # GUI application
    ├── audio.py         # Audio analysis (FFT, beat detection)
    ├── renderer.py      # OpenGL rendering
    ├── renderer_gpudirect.py  # GPU-accelerated renderer
    └── exporter.py      # Video encoding
```

## License

MIT License - Feel free to use, modify, and distribute.

## Credits

Built with:
- [ModernGL](https://github.com/moderngl/moderngl) - OpenGL rendering
- [Librosa](https://librosa.org/) - Audio analysis
- [CuPy](https://cupy.dev/) - GPU acceleration
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - Modern GUI
