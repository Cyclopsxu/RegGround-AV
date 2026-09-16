# RegGround-AV v2 实现核验记录

日期：2026-07-17
冻结手册：`docs/RegGround-AV_修订实现手册_v2.0.md`
工作树基线：`929b739`（实现改动尚未提交）

## Gate 0

Gate 0 已在冻结手册中归档。统计口径为 85 个审计项，不外推为 259 场全量总体：

- LLM 裁定 425，uncertain 301；
- benchmark exclude 267：谓词仅覆盖红灯 186、停车关系不明 39、退化 30、GT precheck 失败 12；
- rule engine veto 71，全部来自旧停止线代理谓词；
- `audit_026` 位移约 0.0 m、静止 5.95 s，旧实现仍给出 veto；
- A11 五条中四条为无证据细节，一条 `audit_029` 为忽略已有证据。

## 实现范围

- Layer 1：scope cut、短窗口标记、转向意图、本向停止线、前向横道、分 kind 冲突区、智能体 track 交互、场景相关注入与可见特征碰撞重试；
- Layer 2：生产图从 6 个 condition / 8 条 rule / 1 个 override，调整为 5 / 7 / 0；测试 fixture 保留 override 机制；
- Layer 3：`SceneFactsDigest`、中性渲染、prompt 要件小节、共享谓词、可配置 2.0 s 相位有效期；
- Benchmark / scoring：override 后置打标、typed `exclude_reason`、候选级 exclude、pair 指标、scenario × difficulty 分层与 `decided_by` 计数。

## Gate 1

手算 fixture 已自动化并通过：

- F1：agent first，gap = +1.5 s；
- F2：temporal overlap，gap < 0；
- F3：对向路径共同区域分别计算双方到达时刻；
- F4：signed distance = -2.0 m，谓词 NOT_DECIDABLE；
- F5：累计航向 -60°，turn intent = right。

同时覆盖 track 截断、最多三个智能体、右转豁免、相位有效期、不可注入标记、digest 隔离与 override 后置打标。

## 自动检查

- `pytest -q`：318 passed；
- `ruff check src tests`：通过；
- `mypy --explicit-package-bases src`：51 个源文件零错误；
- `git diff --check`：通过；
- 全仓覆盖率：74.76%，未达到仓库配置的 85% 门槛。缺口主要位于本次未修改的 CLI、geometry audit、availability 和 experiments；未通过修改 omit 范围规避该失败。

## 实验门状态

- Gate 2：已生成 10 场景、60 张候选轨迹图及逐场景特征清单，等待人工核对；
- Gate 3：20 场景真实 LLM judge 冒烟；
- 全量 v2 重跑、盲标包 v2、IAA/gold 与消融。

这些步骤依赖人工视觉判断、外部 LLM 调用或全量实验输出，不计入本次代码实现完成状态，也不得在未执行时标记通过。
