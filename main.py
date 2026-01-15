#!/usr/bin/env python3
"""
VisualiserTool - Audio-reactive music visualizer

Usage:
    python main.py <audio_file> [options]

Examples:
    python main.py track.mp3
    python main.py track.mp3 --preset universe_within --resolution 3840x2160
    python main.py track.mp3 -r 1080x1920 -o vertical.mp4
"""

from src.cli import main

if __name__ == '__main__':
    main()
