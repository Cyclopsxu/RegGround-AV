# RegGround-AV 点火前修复包 · 2026-07-16

> 依据:10 单元试跑取证(index 4 超时解剖:应用层 2 次 × SDK 隐藏 3 次 = 186s;stage 级降级连坐 6 条;成功延迟分布在 30s 处右删失)
> 定位:全量 259 单元点火前的**最后一个**变更包。一次提交、一次复烟、然后点火。

---

## 一条红线(先于一切改动)

**uncertain 分两种,只有一种允许重跑:**

| 类型 | 标识 | 性质 | 可否重跑 |
|---|---|---|---|
| 语义 uncertain | `decided_by: llm` | 裁判看了、答了"信息不足"——这是一个**判决** | **永不**。重跑到改口 = 给自己的裁判 p-hacking |
| 基础设施 uncertain | `decided_by: fallback` | 压根没得到回答——这是**数据缺失** | 可以。补跑是把没做完的评估做完 |

资格判定在代码里焊死(按 `decided_by` 来源),不留人为裁量。

---

## 修复项

### 1. 关闭 SDK 隐藏重试
- `llm_client.py` 创建 OpenAI client 时显式 `max_retries=0`。
- 理由:重试所有权收归应用层(那里有 telemetry 计数);消灭 2×3 嵌套放大(186s 案根因)。
- **预期副作用,现在立好**:原本被 SDK 暗中救活的调用将显性失败,可见重试数上升——**是诚实化,不是回归**,复烟时勿误读。

### 2. 升级路径预算放宽
- hard_filter 升级调用:`timeout 30s → 90s`,应用层 `retries=2`;pairwise / label_generation 维持原紧超时(P50 仅 2.4s / 6.6s,不动)。
- 理由:成功延迟分布在 30s 处右删失,真实上限未知(成功 stage 已见 82.5s);该路径调用极稀(mini 全批 1 次),90s 总代价 ≈ 0。
- 预算语义写进注释:**逻辑调用总上限 = (retries+1) × timeout**,本次的语义错位不再犯。

### 3. 降级粒度:stage 级 → 候选级
- 规则引擎的确定性判定**保留**;仅升级失败的那条候选落 pending(见第 4 项),其余照常。
- 理由:index 4 一次调用失败连坐 6 条(含规则引擎 confidence 1.0 的判定被清空);方向 fail-closed 没错,但把局部失败放大成整单元弃权,伤数据产出率。
- 实现要点:try/except 从 stage 包裹挪到 per-candidate。

### 4. 清扫轮(deferred repair pass,"期末补考制")
```
主循环:逻辑调用耗尽预算 → 候选记 pending(不落 uncertain)→ 继续
主循环结束 → 清扫轮:只补跑 pending 调用(候选级,不重跑单元)
  成功 → 正常 verdict 入库,标 completed_by: deferred_retry + 各次尝试时间戳
  N=3 轮(轮间隔拉开)仍失败 → 此时才落 fallback/uncertain
```
- 天然吃掉 episode 型故障(重连窗口、限流窗口)——窗口过去补跑必成,现行机制却在窗口内一次失败即永久定案。
- 四条约束焊死:同 run 会话 / 同 commit / 同 manifest 内完成(禁"明天补一批合并");轮数有界;资格 = 仅 `decided_by: fallback` 来源(红线);全程留痕(telemetry:`deferred_attempts` / `recovered_count`)。

### 5. 基础设施加固(试跑遗留三件)
- 连接级事件日志:记录重连/连接异常的**发生时刻**(本次取证只能拿重复 record 当代理时钟,盲区补上);
- 写入幂等:按 unit id 判重,同 unit 不写第二条——宿主重连的重复索引问题从此不需要手工去重规则;
- 进程放 tmux/nohup;(可选)unit 级断点续跑——3 小时 run 死在第 200 单元时的保险。

---

## 验收线(预注册,更新版)

- **清扫后 degraded ≈ 0**;degraded 单元损失率上限 2% 保留(注意:候选级降级后,degraded 不再等于丢单元,"单元损失"仅指 no_choosable_candidate 由降级导致的情形);
- 所有 degraded 均 fail-safe 落 uncertain(fail-open 零容忍);
- **语义 uncertain 率如实报告、不设目标**——裁判正常弃权不是 KPI,压它 = 奖励瞎判;
- degraded 时间聚集 → 判基础设施事故 → 查因 → 整段重跑。

---

## 复烟清单(10 单元,提交后)

- [ ] 全部改动一次提交,clean tree;
- [ ] 10/10 ok;degraded 为 0,或有 degraded 但**规则引擎判定存活**(候选级生效的直接证据);
- [ ] **故障注入一次**:人为掐断一条升级调用 → 确认进 pending → 清扫轮救回 → `completed_by: deferred_retry` 留痕完整;
- [ ] 可见重试数上升属预期(第 1 项副作用),勿当回归;
- [ ] smoke 日志 commit 入库(检查点证据文化)。

## 复烟通过 → 点火

- 全量 **259 单元全跑**(含试跑用过的 10 个;分析只认全量输出,试跑作废不拼接);
- tmux 挂一晚,盯 telemetry:429 / degraded / deferred_recovered 三个计数;
- 中途 bug 铁律不变:修 → 提交 → **整段从头重跑**;
- 跑完 → 首轮分析按冻结口径(illegal-chosen 主指标 / 分层 verdict / unscorable 单列 / degraded 与语义 uncertain 如实分列)。
