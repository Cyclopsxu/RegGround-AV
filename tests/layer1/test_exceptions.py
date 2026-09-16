"""Layer 1 异常层级测试。"""

import pytest

from src.layer1.exceptions import (
    DatasetNotFoundError,
    Layer1Error,
    SceneNotFoundError,
    SceneParseError,
    SceneValidationError,
    TrajectoryAnalysisError,
    TrajectoryAugmentationError,
    TrajectoryExtractionError,
)


class TestExceptionHierarchy:
    """验证异常继承链与基本行为。"""

    def test_all_exceptions_inherit_from_layer1_error(self):
        """所有自定义异常必须是 Layer1Error 的子类。"""
        subclasses = [
            DatasetNotFoundError,
            SceneNotFoundError,
            SceneParseError,
            TrajectoryExtractionError,
            SceneValidationError,
            TrajectoryAugmentationError,
            TrajectoryAnalysisError,
        ]
        for cls in subclasses:
            assert issubclass(cls, Layer1Error), f"{cls.__name__} 应继承 Layer1Error"

    def test_layer1_error_is_standard_exception(self):
        """Layer1Error 应是标准 Exception 的子类。"""
        assert issubclass(Layer1Error, Exception)

    def test_can_catch_by_base_class(self):
        """子类异常可被 Layer1Error 捕获。"""
        for exc_cls in [
            DatasetNotFoundError,
            SceneNotFoundError,
            SceneParseError,
        ]:
            with pytest.raises(Layer1Error):
                raise exc_cls("test")

    def test_can_catch_specifically(self):
        """可按具体异常类型分别捕获。"""
        with pytest.raises(DatasetNotFoundError):
            raise DatasetNotFoundError("data not found")

        with pytest.raises(SceneNotFoundError):
            raise SceneNotFoundError("scene not found")

    def test_exception_message_preserved(self):
        """异常消息应正确传递。"""
        msg = "nuScenes 根目录不存在: /fake/path"
        exc = DatasetNotFoundError(msg)
        assert str(exc) == msg
