"""Layer 2 自定义异常。

本模块定义图谱检索层的所有异常类型。相比 v1.0 大幅精简：
删除了全部 Cypher / LLM / Neo4j 连接相关异常。
"""


class Layer2Error(Exception):
    """Layer 2 所有自定义异常的基类。"""


class GraphDefinitionError(Layer2Error):
    """YAML 图定义文件不存在或无法解析。

    Fatal 级异常，在图加载阶段（RuleGraph.load()）抛出，
    调用方应终止启动或降级为告警模式。
    """


class RuleGraphValidationError(Layer2Error):
    """图完整性校验失败。

    包括 id 重复、悬空边、非法 severity/category 值等。
    Fatal 级异常，视为配置错误，必须修复图定义文件后重新启动。
    """
