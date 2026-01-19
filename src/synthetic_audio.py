"""Synthetic audio generator for real-time shader preview."""

import numpy as np
import math


class SyntheticAudioGenerator:
    """Generates fake audio data using sin waves for shader preview.

    Produces bass, mids, highs, beat, and envelope values that simulate
    music-like patterns without needing an actual audio file.
    """

    def __init__(self, fps: int = 20):
        self.fps = fps
        self.frame = 0

        # Envelope follower state (for AR envelope computation)
        self.envelope_state = 0.0
        self.bass_env_state = 0.0
        self.transient_state = 0.0
        self.sustain_state = 0.0

        # Default envelope time constants (in ms)
        self.attack_ms = 5.0
        self.release_ms = 150.0

    def set_envelope_params(self, attack_ms: float, release_ms: float):
        """Update envelope attack/release times."""
        self.attack_ms = max(1.0, attack_ms)
        self.release_ms = max(20.0, release_ms)

    def _compute_envelope_sample(self, signal: float, state: float,
                                  attack_ms: float, release_ms: float) -> float:
        """Compute one sample of AR envelope follower."""
        attack_coef = math.exp(-1.0 / (self.fps * attack_ms / 1000.0))
        release_coef = math.exp(-1.0 / (self.fps * release_ms / 1000.0))

        if signal > state:
            coef = attack_coef
        else:
            coef = release_coef

        return coef * state + (1.0 - coef) * signal

    def get_frame_data(self, multipliers: dict = None) -> dict:
        """Generate synthetic audio data for the current frame.

        Args:
            multipliers: Optional dict with keys 'bass', 'mids', 'highs', 'beat', 'speed'
                        Values are multipliers (default 1.0 for each)

        Returns:
            Dict with same structure as AudioAnalyzer.get_frame_data()
        """
        if multipliers is None:
            multipliers = {}

        bass_mult = multipliers.get('bass', 1.0)
        mids_mult = multipliers.get('mids', 1.0)
        highs_mult = multipliers.get('highs', 1.0)
        beat_mult = multipliers.get('beat', 1.0)
        speed_mult = multipliers.get('speed', 1.0)

        t = self.frame / self.fps

        # Generate synthetic signals using sin waves
        # Bass: slow pulse ~1 Hz with some variation
        bass_raw = (math.sin(t * 2 * math.pi * 1.0) * 0.5 + 0.5) * 0.7
        bass_raw += (math.sin(t * 2 * math.pi * 0.5) * 0.5 + 0.5) * 0.3

        # Mids: medium frequency ~2 Hz
        mids_raw = (math.sin(t * 2 * math.pi * 2.0) * 0.5 + 0.5) * 0.6
        mids_raw += (math.sin(t * 2 * math.pi * 1.3) * 0.5 + 0.5) * 0.4

        # Highs: faster ~4 Hz with some shimmer
        highs_raw = (math.sin(t * 2 * math.pi * 4.0) * 0.5 + 0.5) * 0.5
        highs_raw += (math.sin(t * 2 * math.pi * 6.0) * 0.5 + 0.5) * 0.3
        highs_raw += (math.sin(t * 2 * math.pi * 8.0) * 0.5 + 0.5) * 0.2

        # Beat: periodic spike every ~0.5 seconds (120 BPM)
        beat_phase = (t * 2.0) % 1.0  # 0 to 1 twice per second
        beat_raw = max(0, 1.0 - beat_phase * 8.0)  # Sharp decay from spike

        # Apply multipliers
        bass = min(1.0, bass_raw * bass_mult)
        mids = min(1.0, mids_raw * mids_mult)
        highs = min(1.0, highs_raw * highs_mult)
        beat = min(1.0, beat_raw * beat_mult)
        volume = (bass + mids + highs) / 3.0

        # Compute combined signal for envelopes
        combined = bass * 0.4 + mids * 0.3 + highs * 0.2 + beat * 0.1

        # Update envelope followers
        self.envelope_state = self._compute_envelope_sample(
            combined, self.envelope_state, self.attack_ms, self.release_ms
        )
        self.bass_env_state = self._compute_envelope_sample(
            bass, self.bass_env_state, 10.0, 200.0
        )
        self.transient_state = self._compute_envelope_sample(
            combined, self.transient_state, 2.0, 50.0
        )
        self.sustain_state = self._compute_envelope_sample(
            combined, self.sustain_state, 50.0, 300.0
        )

        # Audio time accumulates based on energy and speed multiplier
        audio_time = t * speed_mult * (0.5 + combined * 0.5)

        self.frame += 1

        return {
            # Raw frequency bands
            'bass': bass,
            'mids': mids,
            'highs': highs,
            'volume': volume,
            'beat': beat,
            # Time values
            'time': t,
            'audio_time': audio_time,
            # Envelope followers
            'envelope': self.envelope_state,
            'bass_env': self.bass_env_state,
            'transient': self.transient_state,
            'sustain': self.sustain_state,
        }

    def reset(self):
        """Reset the generator to initial state."""
        self.frame = 0
        self.envelope_state = 0.0
        self.bass_env_state = 0.0
        self.transient_state = 0.0
        self.sustain_state = 0.0
