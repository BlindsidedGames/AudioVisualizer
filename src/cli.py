"""Command-line interface for the visualizer tool."""

import click
from pathlib import Path


def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parse resolution string like '1920x1080' into (width, height)."""
    parts = res_str.lower().split('x')
    if len(parts) != 2:
        raise click.BadParameter(f"Invalid resolution format: {res_str}. Use WIDTHxHEIGHT (e.g., 1920x1080)")
    return int(parts[0]), int(parts[1])


@click.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--preset', '-p', default='universe_within', help='Visualization preset name')
@click.option('--resolution', '-r', default='1920x1080', help='Output resolution (e.g., 1920x1080, 3840x2160)')
@click.option('--output', '-o', default=None, help='Output video path (default: audio_visualized.mp4)')
@click.option('--fps', default=60, help='Frames per second')
@click.option('--codec', type=click.Choice(['h264', 'hevc']), default='h264', help='Video codec')
@click.option('--no-nvenc', is_flag=True, help='Disable NVIDIA hardware encoding')
@click.option('--bitrate', default='10M', help='Video bitrate (e.g., 10M, 20M)')
@click.option('--preview', is_flag=True, help='Preview 10 seconds only')
@click.option('--start', default=0, type=float, help='Start time in seconds (skip intro)')
@click.option('--duration', default=None, type=float, help='Duration in seconds (default: full track)')
@click.option('--fast', is_flag=True, help='Use threaded renderer for better performance')
@click.option('--cuda', is_flag=True, help='Use CUDA-optimized renderer with PyTorch')
@click.option('--gpudirect', is_flag=True, help='Use GPU-direct renderer with CuPy (requires CUDA 12)')
def main(audio_file, preset, resolution, output, fps, codec, no_nvenc, bitrate, preview, start, duration, fast, cuda, gpudirect):
    """
    Render an audio-reactive visualization video.

    AUDIO_FILE: Path to the input audio file (MP3, WAV, etc.)

    Examples:
        python main.py track.mp3
        python main.py track.mp3 --preset universe_within --resolution 3840x2160
        python main.py track.mp3 -r 1080x1920 -o vertical_video.mp4
    """
    from .audio import AudioAnalyzer

    # Choose renderer and exporter based on flags
    renderer_mode = None

    if gpudirect:
        # GPU-direct path with CuPy
        try:
            from .renderer_gpudirect import GPUDirectRenderer as ShaderRenderer
            from .renderer_gpudirect import GPUDirectExporter as VideoExporter
            SoftwareVideoExporter = VideoExporter
            renderer_mode = "gpudirect"
            click.echo("Using GPU-direct renderer with CuPy")
        except ImportError as e:
            click.echo(f"GPU-direct not available ({e}), falling back to fast renderer")
            gpudirect = False
            fast = True

    if cuda and not gpudirect:
        # CUDA-optimized path with PyTorch
        try:
            import torch  # Load CUDA runtime
            from .renderer_cuda import CudaShaderRenderer as ShaderRenderer
            from .renderer_cuda import HybridExporter as VideoExporter
            SoftwareVideoExporter = VideoExporter
            renderer_mode = "cuda"
            click.echo("Using CUDA-optimized renderer")
        except ImportError as e:
            click.echo(f"CUDA not available ({e}), falling back to fast renderer")
            cuda = False
            fast = True

    if fast and not cuda and not gpudirect:
        from .renderer_fast import FastShaderRenderer as ShaderRenderer
        from .renderer_fast import ThreadedExporter as VideoExporter
        SoftwareVideoExporter = VideoExporter
        renderer_mode = "fast"
        click.echo("Using fast threaded renderer")

    if renderer_mode is None:
        from .renderer import ShaderRenderer
        from .exporter import VideoExporter, SoftwareVideoExporter
        renderer_mode = "standard"

    # Parse resolution
    width, height = parse_resolution(resolution)
    click.echo(f"Resolution: {width}x{height}")

    # Set output path in outputs folder
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    if output is None:
        audio_name = Path(audio_file).stem
        output = str(output_dir / f"{audio_name}_visualized.mp4")
    elif not Path(output).is_absolute():
        output = str(output_dir / output)

    # Find shader file
    shader_dir = Path(__file__).parent.parent / "shaders"
    shader_path = shader_dir / f"{preset}.glsl"
    if not shader_path.exists():
        available = [f.stem for f in shader_dir.glob("*.glsl")]
        raise click.ClickException(f"Preset '{preset}' not found. Available: {available}")

    click.echo(f"Using preset: {preset}")
    click.echo(f"Output: {output}")

    # Load and analyze audio
    click.echo("\nAnalyzing audio...")
    audio = AudioAnalyzer(audio_file, fps=fps)
    audio.load()

    # Calculate frame range
    start_frame = int(start * fps)
    if start_frame >= audio.frame_count:
        raise click.ClickException(f"Start time {start}s exceeds audio duration {audio.duration:.1f}s")

    # Determine end frame
    if duration is not None:
        end_frame = min(start_frame + int(duration * fps), audio.frame_count)
    elif preview:
        end_frame = min(start_frame + fps * 10, audio.frame_count)
    else:
        end_frame = audio.frame_count

    total_frames = end_frame - start_frame
    click.echo(f"Rendering: {start:.1f}s to {end_frame / fps:.1f}s ({total_frames} frames)")

    # Initialize renderer
    click.echo("\nInitializing renderer...")
    renderer = ShaderRenderer(width, height)
    renderer.load_shader(str(shader_path))

    # Initialize exporter
    if renderer_mode in ("gpudirect", "cuda", "fast"):
        exporter = VideoExporter(
            output_path=output,
            width=width,
            height=height,
            fps=fps,
            audio_path=audio_file,
            codec=codec,
            use_nvenc=not no_nvenc,
            bitrate=bitrate,
        )
    else:
        ExporterClass = SoftwareVideoExporter if no_nvenc else VideoExporter
        exporter = ExporterClass(
            output_path=output,
            width=width,
            height=height,
            fps=fps,
            audio_path=audio_file,
            codec=codec,
            bitrate=bitrate,
        )

    # Render frames
    click.echo(f"\nRendering {total_frames} frames...")
    exporter.start()

    import sys
    import time as time_module
    start_time = time_module.time()

    for i, frame in enumerate(range(start_frame, end_frame)):
        audio_data = audio.get_frame_data(frame)
        frame_time = audio_data['time']

        if renderer_mode in ("gpudirect", "cuda"):
            pixels = renderer.render_frame(frame_time, frame, audio_data)
        elif renderer_mode == "fast":
            pixels = renderer.render_and_get_frame(frame_time, frame, audio_data)
        else:
            pixels = renderer.render_frame_rgb(frame_time, frame, audio_data)

        exporter.write_frame(pixels)

        # Progress update every 60 frames (1 second)
        if i % 60 == 0 or i == total_frames - 1:
            pct = (i + 1) / total_frames * 100
            elapsed = time_module.time() - start_time
            fps_actual = (i + 1) / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\r  Progress: {i + 1}/{total_frames} ({pct:.1f}%) - {fps_actual:.1f} fps")
            sys.stdout.flush()

    print()  # Newline after progress

    # Finish up
    success = exporter.finish()
    renderer.close()

    if success:
        click.echo(f"\nDone! Video saved to: {output}")
    else:
        if not no_nvenc:
            click.echo("\nNVENC failed. Try running with --no-nvenc for software encoding.")
        raise click.ClickException("Video encoding failed")


if __name__ == '__main__':
    main()
