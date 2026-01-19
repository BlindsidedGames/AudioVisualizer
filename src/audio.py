"""Audio analysis module - extracts frequency bands from audio files."""

import numpy as np
import librosa


class AudioAnalyzer:
    """Analyzes audio files and extracts frequency band data per frame."""

    def __init__(self, audio_path: str, fps: int = 60):
        self.audio_path = audio_path
        self.fps = fps
        self.y = None  # Audio time series
        self.sr = None  # Sample rate
        self.duration = None
        self.frame_count = None
        self.hop_length = None

        # Frequency band data arrays
        self.bass = None
        self.mids = None
        self.highs = None
        self.volume = None
        self.beat = None  # Beat detection channel

        # Envelope follower data (AR envelope with separate attack/release)
        self.envelope = None    # Master envelope (all bands)
        self.bass_env = None    # Bass-only envelope
        self.transient = None   # Fast attack, fast release (catches hits)
        self.sustain = None     # Slow attack, slow release (captures held energy)

    def load(self):
        """Load audio file and compute frequency bands for each frame."""
        print(f"Loading audio: {self.audio_path}")
        self.y, self.sr = librosa.load(self.audio_path, sr=None, mono=True)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        self.frame_count = int(self.duration * self.fps)
        self.hop_length = len(self.y) // self.frame_count

        print(f"Duration: {self.duration:.2f}s, Frames: {self.frame_count}, Sample rate: {self.sr}")

        self._compute_frequency_bands()

    def _compute_frequency_bands(self):
        """Compute bass, mids, highs, volume and beat for each frame."""
        print("Computing frequency bands...")

        # Compute STFT with smaller hop for better time resolution
        n_fft = 2048
        hop = self.hop_length
        stft = np.abs(librosa.stft(self.y, n_fft=n_fft, hop_length=hop))

        # Frequency bins
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)

        # Define frequency ranges (Hz)
        bass_range = (20, 250)
        mids_range = (250, 4000)
        highs_range = (4000, 16000)

        # Get bin indices for each range
        bass_bins = np.where((freqs >= bass_range[0]) & (freqs < bass_range[1]))[0]
        mids_bins = np.where((freqs >= mids_range[0]) & (freqs < mids_range[1]))[0]
        highs_bins = np.where((freqs >= highs_range[0]) & (freqs < highs_range[1]))[0]

        # Sum energy in each band per frame
        bass_raw = np.sum(stft[bass_bins, :], axis=0)
        mids_raw = np.sum(stft[mids_bins, :], axis=0)
        highs_raw = np.sum(stft[highs_bins, :], axis=0)
        volume_raw = np.sum(stft, axis=0)

        # Normalize using percentile to preserve dynamics better
        self.bass = self._normalize_dynamic(bass_raw)
        self.mids = self._normalize_dynamic(mids_raw)
        self.highs = self._normalize_dynamic(highs_raw)
        self.volume = self._normalize_dynamic(volume_raw)

        # Resample to exact frame count
        self.bass = self._resample(self.bass, self.frame_count)
        self.mids = self._resample(self.mids, self.frame_count)
        self.highs = self._resample(self.highs, self.frame_count)
        self.volume = self._resample(self.volume, self.frame_count)

        # Light smoothing - keeps reactivity while reducing jitter
        self.bass = self._smooth(self.bass, window_size=3)
        self.mids = self._smooth(self.mids, window_size=3)
        self.highs = self._smooth(self.highs, window_size=3)
        self.volume = self._smooth(self.volume, window_size=3)

        # Detect beats/onsets for punchy response
        print("Detecting beats...")
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=hop)
        onset_env = self._normalize_dynamic(onset_env)
        self.beat = self._resample(onset_env, self.frame_count)
        # Minimal smoothing for beats - keep them punchy
        self.beat = self._smooth(self.beat, window_size=2)

        # Compute accumulated "audio time" that speeds up/slows down with energy
        # This creates smooth acceleration/deceleration instead of jumps
        print("Computing audio time...")
        self._compute_audio_time()

        # Compute envelope followers with different time characteristics
        print("Computing envelopes...")
        self._compute_envelopes()

        print("Frequency bands computed.")

    def _compute_audio_time(self):
        """Compute accumulated time that varies with audio energy."""
        dt = 1.0 / self.fps  # Time step per frame

        # Speed varies with ALL audio energy - bass, mids, highs, and beats
        # This catches melodic elements like "boop beep" sounds in mids/highs
        # Raw speed - shader presets apply their own multipliers
        speed = 0.3 + self.bass * 1.5 + self.mids * 2.0 + self.highs * 1.0 + self.beat * 1.0

        # Light smoothing to keep it responsive
        speed = self._smooth(speed, window_size=2)

        # Accumulate time
        self.audio_time = np.zeros(self.frame_count)
        accumulated = 0.0
        for i in range(self.frame_count):
            accumulated += dt * speed[i]
            self.audio_time[i] = accumulated

    def _compute_envelopes(self):
        """
        Compute envelope followers with different attack/release characteristics.

        Creates four envelope types:
        - envelope: Master AR envelope (all bands summed), attack 5ms, release 150ms
        - bass_env: Bass-only envelope, attack 10ms, release 200ms (slower for sub feel)
        - transient: Fast attack/release for punchy hits, attack 2ms, release 50ms
        - sustain: Slow attack/release for held energy, attack 50ms, release 300ms
        """
        # Combined signal for master envelope (weighted sum)
        combined = self.bass * 0.4 + self.mids * 0.3 + self.highs * 0.2 + self.beat * 0.1

        # Master envelope: balanced attack/release
        self.envelope = self._compute_envelope(combined, attack_ms=5, release_ms=150)

        # Bass envelope: slower for that sub feel
        self.bass_env = self._compute_envelope(self.bass, attack_ms=10, release_ms=200)

        # Transient: fast attack, fast release - catches individual hits
        self.transient = self._compute_envelope(combined, attack_ms=2, release_ms=50)

        # Sustain: slow attack, slow release - captures held energy, ignores transients
        self.sustain = self._compute_envelope(combined, attack_ms=50, release_ms=300)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize array to 0-1 range."""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val == 0:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    def _normalize_dynamic(self, data: np.ndarray) -> np.ndarray:
        """Normalize using percentile to preserve dynamics and punch."""
        # Use 5th percentile as floor (ignores silence)
        # Use 95th percentile as ceiling (prevents outliers from squashing everything)
        p5 = np.percentile(data, 5)
        p95 = np.percentile(data, 95)

        if p95 - p5 == 0:
            return np.zeros_like(data)

        normalized = (data - p5) / (p95 - p5)
        # Clip to 0-1 but allow some headroom for peaks
        return np.clip(normalized, 0, 1.5) / 1.5

    def _resample(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """Resample data to target length."""
        indices = np.linspace(0, len(data) - 1, target_length)
        return np.interp(indices, np.arange(len(data)), data)

    def _smooth(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """Apply exponential moving average smoothing for natural feel."""
        # Use larger window for smoother transitions
        alpha = 2.0 / (window_size + 1)
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed

    def _compute_envelope(self, signal: np.ndarray, attack_ms: float = 5, release_ms: float = 150) -> np.ndarray:
        """
        Compute envelope with separate attack/release times.
        Similar to a compressor's detector circuit.

        Args:
            signal: Input signal (0-1 normalized)
            attack_ms: Attack time in milliseconds (how fast it rises)
            release_ms: Release time in milliseconds (how fast it falls)

        Returns:
            Envelope follower output (0-1)
        """
        # Convert ms to coefficient (time constant)
        # coef = exp(-1 / (fps * time_in_seconds))
        attack_coef = np.exp(-1.0 / (self.fps * attack_ms / 1000.0))
        release_coef = np.exp(-1.0 / (self.fps * release_ms / 1000.0))

        envelope = np.zeros_like(signal)
        envelope[0] = signal[0]

        for i in range(1, len(signal)):
            if signal[i] > envelope[i-1]:
                # Attack - fast rise (use attack coefficient)
                coef = attack_coef
            else:
                # Release - slow decay (use release coefficient)
                coef = release_coef
            envelope[i] = coef * envelope[i-1] + (1.0 - coef) * signal[i]

        return envelope

    def get_frame_data(self, frame: int) -> dict:
        """Get audio data for a specific frame."""
        if frame >= self.frame_count:
            frame = self.frame_count - 1
        return {
            # Raw frequency bands
            'bass': float(self.bass[frame]),
            'mids': float(self.mids[frame]),
            'highs': float(self.highs[frame]),
            'volume': float(self.volume[frame]),
            'beat': float(self.beat[frame]),
            # Time values
            'time': frame / self.fps,
            'audio_time': float(self.audio_time[frame]),
            # Envelope followers (AR envelopes with proper attack/release)
            'envelope': float(self.envelope[frame]),      # Master envelope
            'bass_env': float(self.bass_env[frame]),      # Bass-only envelope
            'transient': float(self.transient[frame]),    # Fast attack/release
            'sustain': float(self.sustain[frame]),        # Slow attack/release
        }
