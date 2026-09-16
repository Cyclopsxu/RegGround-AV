"""轨迹特征在 LLM 提示词中的唯一文本表示。"""

from __future__ import annotations

import hashlib

from src.layer1.models import TrajectoryFeatures


def render_feature_text(feature: TrajectoryFeatures) -> str:
    """按生产提示词的舍入精度渲染轨迹特征。"""
    return (
        f"{feature.natural_language_summary}；"
        f"速度 max={feature.max_speed_mps:.2f}, "
        f"min={feature.min_speed_mps:.2f}, mean={feature.mean_speed_mps:.2f}；"
        f"冲突区行为: {feature.conflict_zone_behavior or '无'}"
    )


def feature_text_hash(feature: TrajectoryFeatures) -> str:
    """返回 LLM 实际可见文本的稳定 SHA-256。"""
    return hashlib.sha256(render_feature_text(feature).encode("utf-8")).hexdigest()
