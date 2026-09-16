# RegGround-AV 实验配置 v2.0

本表是主实验与后续 2×2 消融的候选运行参数；Gate 3 复冒烟通过后才冻结。四个消融条件只允许切换“规则检索”和“引用校验”两个目标开关；下列模型、温度、超时、重试、上下文与候选生成参数必须完全相同。

## Layer 3 / DeepSeek

| 参数 | 候选值 |
|---|---|
| provider | `openai_compatible` |
| API base URL | `https://api.deepseek.com` |
| model alias | `deepseek-v4-pro` |
| provider model version | `DeepSeek-V4-Pro` |
| thinking | `enabled` |
| hard-filter temperature | `0.0` |
| pairwise temperature | `0.0` |
| label temperature | `0.2` |
| structured max tokens | `8192` |
| text max tokens | `4096` |
| general timeout | `180s` |
| hard-filter timeout | `180s` |
| general retries | `1` |
| hard-filter retries | `2` |
| retry backoff | 随机 `3–6s × 4^attempt` |
| deferred retry | 3 轮，轮间 5s |
| prompt budget | 8000 tokens |
| max trajectories | 10 |
| max rules | 20 |
| max pairwise comparisons | 45 |

每次正式日志 manifest 必须写入以上字段、prompt SHA、规则图 SHA、共享谓词 SHA、benchmark builder SHA、输入 SHA、冻结 eval-set SHA、git commit 和运行时间。API key 及 reasoning 正文不得入日志。

## Layer 1

| 参数 | 候选值 |
|---|---|
| seed | `42` |
| candidate horizon | `6s` |
| path horizon | `12s` |
| max lateral acceleration | `3.5m/s²` |
| red compliant stop buffer | `1.0m` |
| pedestrian illegal gap target | `-0.3s` |
| stop-line penetration epsilon | `0.15m` |
| drivable-area gate | 开启 |

## Gate 3 覆盖率基线记录

先按结构事实拆成两个池：设计内 oncoming、`start_beyond_line`、pedestrian 触发/交互缺失及 `not_evaluable_short_window` 属于结构不可评分池，其余属于结构可评分池。可评分 pair 定义为同一场景的 benchmark 标签同时至少存在一条 `expected_verdict=cleared` 和一条 `expected_verdict=vetoed`。全量同时报告两个池的场景数、pair 数与比率；该比率只作版本基线记录，不设阈值，也不作通过或未通过判定。20 场景 smoke 只裁行为门禁。禁止通过伪造违规标签或移动 borderline/注入失败场景提高数字。
