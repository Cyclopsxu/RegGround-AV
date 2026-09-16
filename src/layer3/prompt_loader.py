"""Prompt 模板加载工具。

从 prompts/ 目录加载系统 prompt 文本与 few-shot 示例。
"""

from __future__ import annotations

from pathlib import Path

# prompts/ 目录相对本模块的路径
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# 缓存，避免重复磁盘 I/O
_cache: dict[str, str] = {}


def load_system_prompt(stage: str) -> str:
    """加载指定阶段（hard_filter / preference_ranker / label_generator）的系统 prompt。

    Args:
        stage: 阶段标识（不含 _system.txt 后缀）。

    Returns:
        系统 prompt 文本。
    """
    cache_key = f"{stage}_system"
    if cache_key in _cache:
        return _cache[cache_key]

    file_path = _PROMPTS_DIR / f"{stage}_system.txt"
    text = file_path.read_text(encoding="utf-8") if file_path.exists() else ""

    _cache[cache_key] = text
    return text


def load_few_shot_examples(stage: str) -> str:
    """加载指定阶段的 few-shot 示例，原样嵌入 prompt。

    Args:
        stage: 阶段标识（不含 _examples.json 后缀）。

    Returns:
        few-shot 示例文本（JSON 文件不存在时返回空字符串）。
    """
    cache_key = f"{stage}_fewshot"
    if cache_key in _cache:
        return _cache[cache_key]

    file_path = _PROMPTS_DIR / "few_shots" / f"{stage}_examples.json"
    if not file_path.exists():
        _cache[cache_key] = ""
        return ""

    try:
        text = file_path.read_text(encoding="utf-8").strip()
    except OSError:
        _cache[cache_key] = ""
        return ""

    _cache[cache_key] = text
    return text
