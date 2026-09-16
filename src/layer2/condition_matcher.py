"""SceneQuery → Condition 确定性匹配。"""

from src.condition_mapping import FACT_TO_CONDITION
from src.layer1.models import SceneFacts, SceneQuery
from src.layer2.models import ConditionSpec, RetrievalMode


class ConditionMatcher:
    """将 SceneQuery 确定性地映射为命中的 Condition id。

    纯函数式、引用透明：相同 SceneQuery 必返回相同结果。
    """

    def __init__(
        self,
        condition_specs: dict[str, ConditionSpec],
        *,
        enable_keyword_fallback: bool = True,
    ) -> None:
        """初始化条件匹配器。

        Args:
            condition_specs: {condition_id: ConditionSpec} 映射，由 RuleGraph 提供
            enable_keyword_fallback: 结构匹配无命中时是否尝试关键词兜底
        """
        self._condition_specs = condition_specs
        self._enable_fallback = enable_keyword_fallback
        self._fact_to_condition = self._build_fact_mapping(condition_specs)

    def match(self, query: SceneQuery) -> tuple[list[str], RetrievalMode]:
        """将 SceneQuery 映射为命中的 Condition id。

        Args:
            query: Layer 1 输出的 Layer 2 窄接口

        Returns:
            (命中的 condition_id 列表, 检索模式)
        """
        structured = self._conditions_from_facts(query.scene_facts)
        if structured:
            return structured, RetrievalMode.STRUCTURED

        if self._enable_fallback:
            keyword = self._conditions_from_keywords(query.keywords)
            if keyword:
                return keyword, RetrievalMode.KEYWORD_FALLBACK

        return [], RetrievalMode.STRUCTURED

    @staticmethod
    def _build_fact_mapping(condition_specs: dict[str, ConditionSpec]) -> dict[str, str]:
        """由 YAML fact_keys 与文档固定映射共同生成事实映射。"""
        fact_to_condition = dict(FACT_TO_CONDITION)
        for condition_id, spec in condition_specs.items():
            for fact_key in spec.fact_keys:
                fact_to_condition[fact_key] = condition_id
        return fact_to_condition

    def _conditions_from_facts(self, facts: SceneFacts) -> list[str]:
        """读取 SceneFacts 中为 True 的事实并映射为 Condition。"""
        matched: list[str] = []
        for fact_key, condition_id in self._fact_to_condition.items():
            if condition_id not in self._condition_specs:
                continue
            if getattr(facts, fact_key, False) is True:
                matched.append(condition_id)
        return sorted(set(matched))

    def _conditions_from_keywords(self, keywords: list[str]) -> list[str]:
        """对英文关键词做子串包含匹配。"""
        query_keywords = [kw.strip().lower() for kw in keywords if kw.strip()]
        if not query_keywords:
            return []

        matched: list[str] = []
        for cond_id, spec in self._condition_specs.items():
            condition_keywords = [kw.strip().lower() for kw in spec.keywords if kw.strip()]
            if not condition_keywords:
                continue
            if any(
                query_kw in condition_kw or condition_kw in query_kw
                for query_kw in query_keywords
                for condition_kw in condition_keywords
            ):
                matched.append(cond_id)
        return sorted(set(matched))
