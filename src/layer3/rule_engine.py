"""确定性硬规则引擎。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.compliance_predicates import evaluate_active_rules
from src.layer1.models import TrajectoryFeatures
from src.layer3.models import JudgeContext, VetoStatus


class RuleEngineDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    trajectory_id: str
    status: VetoStatus
    violated_rule_ids: list[str] = Field(default_factory=list)
    reason: str
    is_decisive: bool


class RuleEngine:
    """仅使用已审计几何谓词直裁明确样本。"""

    def __init__(self, phase_validity_horizon_s: float = 2.0) -> None:
        if phase_validity_horizon_s <= 0:
            raise ValueError("phase_validity_horizon_s must be positive")
        self._phase_validity_horizon_s = phase_validity_horizon_s

    def check(
        self,
        features: TrajectoryFeatures,
        context: JudgeContext,
        traj_id: str,
    ) -> RuleEngineDecision:
        if not context.hard_rule_ids:
            return RuleEngineDecision(
                trajectory_id=traj_id,
                status=VetoStatus.CLEARED,
                reason="当前上下文没有 hard rules，规则引擎直接放行",
                is_decisive=True,
            )

        outcome = evaluate_active_rules(
            features,
            context.scene_facts_digest,
            set(context.hard_rule_ids),
            phase_validity_horizon_s=self._phase_validity_horizon_s,
        )
        if not outcome.is_decisive:
            return self._uncertain(traj_id, f"{outcome.reason}，交由 LLM 判断")
        if outcome.verdict == "cleared":
            return RuleEngineDecision(
                trajectory_id=traj_id,
                status=VetoStatus.CLEARED,
                reason=outcome.reason,
                is_decisive=True,
            )
        return RuleEngineDecision(
            trajectory_id=traj_id,
            status=VetoStatus.VETOED,
            violated_rule_ids=[
                "R-SIG-01" if "R-SIG-01" in context.hard_rule_ids else "R-YLD-01"
            ],
            reason=outcome.reason,
            is_decisive=True,
        )

    @staticmethod
    def _uncertain(traj_id: str, reason: str) -> RuleEngineDecision:
        return RuleEngineDecision(
            trajectory_id=traj_id,
            status=VetoStatus.UNCERTAIN,
            reason=reason,
            is_decisive=False,
        )


class DisabledRuleEngine:
    """rule-engine-off 条件：任何候选都不直裁，全部交给 LLM。

    影子评估与 2×2 消融的 no_rule_engine 条件共用本实现，保证"关闭规则引擎"
    在两处是同一件事，而不是各写一份近似逻辑。
    """

    DISABLED_REASON = "规则引擎已按配置停用，交由 LLM 判断"

    def check(
        self,
        features: TrajectoryFeatures,
        context: JudgeContext,
        traj_id: str,
    ) -> RuleEngineDecision:
        del features, context
        return RuleEngine._uncertain(traj_id, self.DISABLED_REASON)


def build_rule_engine(
    *,
    phase_validity_horizon_s: float = 2.0,
    disabled: bool = False,
) -> RuleEngine | DisabledRuleEngine:
    """按配置返回生产规则引擎或停用版本。"""
    if disabled:
        return DisabledRuleEngine()
    return RuleEngine(phase_validity_horizon_s)
