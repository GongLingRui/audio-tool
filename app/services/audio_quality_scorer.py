"""
Audio Quality Scorer
Automatically evaluate audio quality for TTS output
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Audio quality metrics."""
    overall_score: float  # 0-100
    clarity: float  # Speech clarity
    naturalness: float  # How natural it sounds
    consistency: float  # Voice consistency
    dynamic_range: float  # Dynamic range utilization
    signal_to_noise: float  # SNR in dB
    artifacts: float  # Presence of artifacts (lower is better)
    recommendation: str  # Human-readable recommendation


class AudioQualityScorer:
    """Score audio quality automatically."""

    def __init__(self):
        self._quality_thresholds = {
            "excellent": 90,
            "good": 75,
            "fair": 60,
            "poor": 40,
        }

    async def score_audio(
        self,
        audio_path: str,
        reference_path: Optional[str] = None,
    ) -> QualityMetrics:
        """
        Score audio quality.

        Args:
            audio_path: Path to audio file to score
            reference_path: Optional reference audio for comparison

        Returns:
            Quality metrics
        """
        try:
            import librosa
            import soundfile as sf
        except ImportError:
            logger.warning("librosa not available, using basic metrics")
            return self._basic_score(audio_path)

        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)

        # Calculate metrics
        clarity = self._calculate_clarity(y, sr)
        naturalness = self._calculate_naturalness(y, sr)
        consistency = self._calculate_consistency(y, sr)
        dynamic_range = self._calculate_dynamic_range(y)
        snr = self._calculate_snr(y)
        artifacts = self._calculate_artifacts(y, sr)

        # Calculate overall score
        overall_score = self._calculate_overall_score({
            "clarity": clarity,
            "naturalness": naturalness,
            "consistency": consistency,
            "dynamic_range": dynamic_range,
            "snr": snr,
            "artifacts": artifacts,
        })

        # Generate recommendation
        recommendation = self._generate_recommendation(overall_score, {
            "clarity": clarity,
            "naturalness": naturalness,
            "consistency": consistency,
            "snr": snr,
        })

        return QualityMetrics(
            overall_score=overall_score,
            clarity=clarity,
            naturalness=naturalness,
            consistency=consistency,
            dynamic_range=dynamic_range,
            signal_to_noise=snr,
            artifacts=artifacts,
            recommendation=recommendation,
        )

    def _calculate_clarity(self, y: np.ndarray, sr: int) -> float:
        """Calculate speech clarity score."""
        # Use MFCCs to measure spectral clarity
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Measure variance in MFCCs (higher variance = more distinct sounds)
        mfcc_var = np.var(mfccs, axis=1)
        clarity = np.mean(mfcc_var)

        # Normalize to 0-100
        return min(100, max(0, clarity * 50))

    def _calculate_naturalness(self, y: np.ndarray, sr: int) -> float:
        """Calculate how natural the speech sounds."""
        # Measure prosodic variation
        # Calculate pitch variation
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if not pitch_values:
            return 50.0  # Neutral score

        pitch_std = np.std(pitch_values)
        pitch_range = np.max(pitch_values) - np.min(pitch_values)

        # Natural speech has moderate variation
        naturalness = min(100, (pitch_std / 50) * 50 + (pitch_range / 200) * 50)
        return naturalness

    def _calculate_consistency(self, y: np.ndarray, sr: int) -> float:
        """Calculate voice consistency."""
        # Measure spectral consistency over time
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        # Lower variation = more consistent
        centroid_std = np.std(spectral_centroids)
        centroid_mean = np.mean(spectral_centroids)

        # Coefficient of variation
        cv = (centroid_std / centroid_mean) if centroid_mean > 0 else 1.0

        # Lower CV = higher consistency
        consistency = max(0, 100 - (cv * 100))
        return consistency

    def _calculate_dynamic_range(self, y: np.ndarray) -> float:
        """Calculate dynamic range utilization."""
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y)[0]

        # Dynamic range in dB
        dr_db = 20 * np.log10(np.max(rms) / (np.min(rms) + 1e-6))

        # Ideal range is 20-30 dB for speech
        if dr_db < 10:
            score = dr_db * 5  # Too compressed
        elif dr_db > 40:
            score = max(0, 100 - (dr_db - 40) * 2)  # Too wide
        else:
            score = 80 + (dr_db - 20) * 2  # Good range

        return min(100, max(0, score))

    def _calculate_snr(self, y: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        # Estimate noise level from quietest portions
        rms = librosa.feature.rms(y=y)[0]

        # Use bottom 10% as noise estimate
        noise_level = np.percentile(rms, 10)
        signal_level = np.percentile(rms, 90)

        if noise_level < 1e-6:
            return 100.0  # Perfect

        snr_db = 20 * np.log10(signal_level / noise_level)

        # Normalize to 0-100 (20+ dB is good)
        return min(100, max(0, snr_db * 3))

    def _calculate_artifacts(self, y: np.ndarray, sr: int) -> float:
        """Calculate presence of artifacts (lower is better)."""
        # Detect clicks, pops, and distortion

        # Zero crossing rate (abnormal spikes)
        zcr = librosa.feature.zero_crossing_rate(y)[0]

        # Detect sudden changes (clicks)
        diff = np.diff(y)
        sudden_changes = np.abs(diff) > np.std(diff) * 5

        # Measure harmonic distortion
        # Use spectral flatness (higher = more noise-like)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        avg_flatness = np.mean(spectral_flatness)

        # Combine metrics (0-100, lower is better)
        artifact_score = (
            np.mean(zcr) * 100 +
            np.mean(sudden_changes) * 50 +
            avg_flatness * 100
        ) / 3

        return min(100, max(0, artifact_score))

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score."""
        weights = {
            "clarity": 0.3,
            "naturalness": 0.25,
            "consistency": 0.2,
            "dynamic_range": 0.1,
            "snr": 0.1,
            "artifacts": -0.05,  # Negative because lower is better
        }

        weighted_sum = 0
        for metric, weight in weights.items():
            value = metrics.get(metric, 0)
            if metric == "artifacts":
                # Invert artifacts (lower is better)
                value = 100 - value
            weighted_sum += value * weight

        return min(100, max(0, weighted_sum))

    def _generate_recommendation(
        self,
        overall_score: float,
        metrics: Dict[str, float],
    ) -> str:
        """Generate human-readable recommendation."""
        if overall_score >= 90:
            return "Excellent quality! No improvements needed."
        elif overall_score >= 75:
            return "Good quality. Minor adjustments may help."
        elif overall_score >= 60:
            issues = []
            if metrics.get("clarity", 0) < 60:
                issues.append("improve clarity with better recording")
            if metrics.get("naturalness", 0) < 60:
                issues.append("add more prosodic variation")
            if metrics.get("snr", 0) < 50:
                issues.append("reduce background noise")

            if issues:
                return f"Fair quality. Consider: {', '.join(issues)}."
            return "Fair quality. Some improvements recommended."
        else:
            return "Poor quality. Significant improvements needed."

    def _basic_score(self, audio_path: str) -> QualityMetrics:
        """Basic scoring without librosa."""
        from pydub import AudioSegment
        from pydub.utils import ratio_to_db

        audio = AudioSegment.from_file(audio_path)

        # Calculate basic metrics
        duration = len(audio) / 1000.0

        # Basic dynamic range
        max_dBFS = audio.max_dBFS
        min_dBFS = audio.min_dBFS
        dynamic_range = max_dBFS - min_dBFS

        # Basic scores
        clarity = min(100, 60 + dynamic_range * 2)
        naturalness = 70.0  # Default
        consistency = 70.0  # Default
        snr = max(0, 50 + dynamic_range)
        artifacts = 20.0  # Default
        overall = (clarity + snr) / 2

        return QualityMetrics(
            overall_score=overall,
            clarity=clarity,
            naturalness=naturalness,
            consistency=consistency,
            dynamic_range=min(100, max(0, dynamic_range * 5)),
            signal_to_noise=snr,
            artifacts=artifacts,
            recommendation="Basic score (full analysis requires librosa)",
        )

    async def batch_score(
        self,
        audio_paths: List[str],
    ) -> List[QualityMetrics]:
        """Score multiple audio files."""
        results = []
        for path in audio_paths:
            try:
                metrics = await self.score_audio(path)
                results.append(metrics)
            except Exception as e:
                logger.error(f"Error scoring {path}: {e}")
                # Add placeholder metrics
                results.append(QualityMetrics(
                    overall_score=0.0,
                    clarity=0.0,
                    naturalness=0.0,
                    consistency=0.0,
                    dynamic_range=0.0,
                    signal_to_noise=0.0,
                    artifacts=100.0,
                    recommendation=f"Error: {str(e)}",
                ))

        return results

    def get_quality_report(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Generate a detailed quality report."""
        return {
            "overall_score": metrics.overall_score,
            "grade": self._score_to_grade(metrics.overall_score),
            "metrics": {
                "clarity": metrics.clarity,
                "naturalness": metrics.naturalness,
                "consistency": metrics.consistency,
                "dynamic_range": metrics.dynamic_range,
                "signal_to_noise": metrics.signal_to_noise,
                "artifacts": metrics.artifacts,
            },
            "recommendation": metrics.recommendation,
            "timestamp": datetime.now().isoformat(),
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"


# Global instance
_quality_scorer: Optional[AudioQualityScorer] = None


def get_audio_quality_scorer() -> AudioQualityScorer:
    """Get global audio quality scorer instance."""
    global _quality_scorer
    if _quality_scorer is None:
        _quality_scorer = AudioQualityScorer()
    return _quality_scorer
