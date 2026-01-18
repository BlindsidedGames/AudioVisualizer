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

        # Apply smoothing to prevent jittery visuals
        # Heavier smoothing (window=8) for smooth transitions
        self.bass = self._smooth(self.bass, window_size=8)
        self.mids = self._smooth(self.mids, window_size=8)
        self.highs = self._smooth(self.highs, window_size=8)
        self.volume = self._smooth(self.volume, window_size=8)

        # Detect beats/onsets for punchy response
        print("Detecting beats...")
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr, hop_length=hop)
        onset_env = self._normalize_dynamic(onset_env)
        self.beat = self._resample(onset_env, self.frame_count)
        # Smooth beats too but lighter so they stay punchy
        self.beat = self._smooth(self.beat, window_size=4)

        # Compute accumulated "audio time" that speeds up/slows down with energy
        # This creates smooth acceleration/deceleration instead of jumps
        print("Computing audio time...")
        self._compute_audio_time()

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

    def get_frame_data(self, frame: int) -> dict:
        """Get audio data for a specific frame."""
        if frame >= self.frame_count:
            frame = self.frame_count - 1
        return {
            'bass': float(self.bass[frame]),
            'mids': float(self.mids[frame]),
            'highs': float(self.highs[frame]),
            'volume': float(self.volume[frame]),
            'beat': float(self.beat[frame]),
            'time': frame / self.fps,
            'audio_time': float(self.audio_time[frame]),
        }
