# Hermes Next v0.4.0 → v0.5.0 — 验证门控版

> 2026-06-13
> 基于 cc-star v0.4.0 验证门控移植，全员确认通过

---

## 新增

### 🆕 promote_gate.py — 验证门控模块

从 cc-star v0.4.0 移植，基于微软 SkillOpt `evaluate_gate()` 纯函数：
- **门控结果** — `reject` / `accept` / `accept_new_best`
- **双指标** — `hard`（policy confidence）/ `soft`（激活频率），默认 `mixed`
- **双线追踪** — `current_score` + `best_score` 持久化
- **拒绝缓冲** — 被 gate 拒绝的 policy 记入 `reject_log.jsonl`

### 🆕 check_trace_freshness() — 新鲜度检查（好二妹建议）

promote 前先验 traces 最新写入时间：
- 最新 trace ≤ 24h → 正常跑 promote + gate
- 最新 trace > 24h → 跳过 promote，log 报警，不消费算力

## 改进

### Policy 晋升加入门控
`PolicyRepository.list_active()` 新增 `use_gate` 参数：
- `True`（默认）→ 候选经 gate 过滤，只返回优于当前的 policy
- `False` → v0.4.0 老逻辑（置信度阈值），不碰 gate_state

## 兼容性

- [x] `HERMES_NEXT_GATE_ENABLED=false` → 行为与 v0.4.0 完全一致
- [x] gate_state.json 仅在门控开启时读写
- [x] 关闭门控后 gate_state 不受影响

## 文件改动

| 文件 | 操作 |
|:----|:-----|
| `hermes_next/memos/promote_gate.py` | 新增 ~170 行 |
| `hermes_next/cache/policies.py` | 修改 ~+30 行 |
| `hermes_next/__init__.py` | 版本号 0.4.0→0.5.0 |
| `pyproject.toml` | 版本号 0.4.0→0.5.0 |

**合计：~200 行新增/修改**

---

**升级方式：** `pip install -e .`
**降级方式：** `export HERMES_NEXT_GATE_ENABLED=false`
