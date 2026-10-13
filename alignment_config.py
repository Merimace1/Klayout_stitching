"""Algorithm parameters, independent of any particular file or machine."""

from dataclasses import dataclass, fields
import json


@dataclass(frozen=True)
class AlignmentConfig:
    letter_height_um: float | None = None
    square_size_um: float | None = None
    target_letter_height_px: float = 12.3
    background_sigma_px: float = 5.0
    contrast_threshold: float = 15.0
    label_score: float = 0.65
    marker_score: float = 0.5
    label_tolerance_um: float = 8.0
    marker_tolerance_um: float = 4.0
    marker_search_px: int = 6
    polarity: str = "dark"
    stitching_min_inliers: int = 5
    stitching_descriptor_ratio: float = 0.95
    stitching_threshold_px: float = 4.0
    stitching_sparse_threshold_px: float = 12.0
    stitching_min_gradient_score: float = 0.25
    stitching_feature_levels: int = 9
    stitching_features_per_level: int = 450
    stitching_max_dimension: int = 1600

    @classmethod
    def from_file(cls, path):
        values = json.loads(path.read_text(encoding="utf-8")) if path else {}
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(
                f"Unknown configuration options: {', '.join(sorted(unknown))}"
            )
        config = cls(**values)
        for field in fields(cls):
            value = getattr(config, field.name)
            if field.name == "polarity":
                if value not in ("dark", "bright"):
                    raise ValueError("polarity must be dark or bright")
            elif value is not None:
                import math

                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"{field.name} must be a positive finite number")
        for name in (
            "marker_search_px",
            "stitching_min_inliers",
            "stitching_feature_levels",
            "stitching_features_per_level",
            "stitching_max_dimension",
        ):
            if not isinstance(getattr(config, name), int):
                raise ValueError(f"{name} must be an integer")
        if config.stitching_min_inliers < 4:
            raise ValueError("stitching_min_inliers must be at least 4")
        if (
            max(
                config.label_score,
                config.marker_score,
                config.stitching_min_gradient_score,
                config.stitching_descriptor_ratio,
            )
            > 1
        ):
            raise ValueError("Correlation score thresholds cannot exceed 1")
        return config
