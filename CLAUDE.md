# VisualiserTool

## Project Overview
A local GPU-accelerated music visualizer tool that renders audio-reactive visualizations to video at any resolution.

## Goals
- **Input**: Audio file (MP3, WAV, etc.)
- **Output**: Video file at arbitrary resolution (1920x1080, 1080x1920, 3840x2160, etc.)
- **Workflow**: CLI-style preset system - select a preset, provide audio, render video automatically
- **Quality**: Professional output with effects like bloom/glow, suitable for music videos

## Tech Stack
- **Language**: Python
- **Audio Analysis**: librosa or numpy FFT for frequency/waveform data
- **Rendering**: ShaderFlow (Shadertoy-compatible GLSL shaders) or moderngl
- **Video Encoding**: FFmpeg with hardware encoding (NVENC for NVIDIA GPUs)
- **Presets**: JSON/YAML config files defining shader + parameters

## Architecture

```
visualiser/
├── presets/           # Preset definitions (YAML/JSON)
│   ├── bars.yaml
│   ├── radial.yaml
│   └── particles.yaml
├── shaders/           # GLSL fragment shaders
│   ├── bars.glsl
│   ├── radial.glsl
│   └── bloom.glsl     # Post-processing
├── src/
│   ├── audio.py       # Audio loading & FFT analysis
│   ├── renderer.py    # GPU rendering pipeline
│   ├── exporter.py    # FFmpeg video encoding
│   └── cli.py         # Command-line interface
├── main.py            # Entry point
└── requirements.txt
```

## CLI Usage (Target)
```bash
python main.py track.mp3 --preset radial --resolution 1920x1080 --output out.mp4
python main.py track.mp3 --preset bars --resolution 3840x2160 --output 4k_video.mp4
```

## Key Components

### Audio Analysis
- Use `librosa` for loading audio and computing FFT
- Extract frequency bands (bass, mids, highs) per frame
- Detect beats/transients for reactive effects

### Shader System
- Shadertoy-compatible GLSL shaders (80% compatible with ShaderFlow)
- Pass audio data as uniforms or textures
- Support post-processing passes (bloom, chromatic aberration)

### Presets
Each preset defines:
- Which shader(s) to use
- Parameter mappings (frequency band -> visual property)
- Resolution-independent settings
- Post-processing chain

### Video Export
- Frame-by-frame rendering to FFmpeg
- Hardware encoding with NVENC (h264_nvenc, hevc_nvenc)
- Proper audio sync with frame timing

## Visual Styles to Support
- **Bar Spectrum**: Classic frequency bars (horizontal/vertical/radial)
- **Waveform**: Line-based waveform display, optionally circular
- **Particles**: VFX-style particles driven by audio
- **Shader Effects**: Raymarching, fractals, displacement, kaleidoscopes
- **Post-Processing**: Bloom/glow, chromatic aberration, color grading

## References
- [ShaderFlow](https://pypi.org/project/shaderflow/) - Python framework for audio-reactive shaders
- [Shadertoy](https://www.shadertoy.com/results?query=visualizer&sort=popular&filter=musicstream) - Shader examples
- [librosa](https://librosa.org/) - Audio analysis library
- [moderngl](https://moderngl.readthedocs.io/) - Python OpenGL bindings

## Development Notes
- Start with ShaderFlow to leverage existing Shadertoy compatibility
- Test with simple bar visualizer first, then add complexity
- Ensure resolution-independence in all shaders
- Profile performance at 4K to identify bottlenecks
