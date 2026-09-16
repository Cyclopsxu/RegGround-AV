"""Layer 1 自定义异常层级。

所有 Layer 1 异常均继承自 Layer1Error，下游可按粒度捕获。
"""


class Layer1Error(Exception):
    """Layer 1 所有自定义异常的基类。"""


class DatasetNotFoundError(Layer1Error):
    """Fatal：nuScenes 根目录或 DriveLM QA 文件不存在，终止流程。"""


class SceneNotFoundError(Layer1Error):
    """Error：scene_token / frame_token 在数据集中不存在。"""


class SceneParseError(Layer1Error):
    """Warning：QA JSON 损坏或字段缺失，单场景解析失败。"""


class TrajectoryExtractionError(Layer1Error):
    """Warning：ego_pose 窗口长度不足以构成轨迹（< min_waypoints）。"""


class SceneValidationError(Layer1Error):
    """Warning：最终 SceneContext 对象违反模型约束。"""


class TrajectoryAugmentationError(Layer1Error):
    """Warning：原始轨迹 waypoint 不足，无法合成候选轨迹。"""


class TrajectoryAnalysisError(Layer1Error):
    """Warning：waypoint 序列过短，无法提取有意义的语义特征。"""
