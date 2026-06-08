# Hermes Next v0.2.1 → v0.3.0 升级指南

> 发布日期：2026-06-08
> 升级范围：好妹、好二妹、灵儿
> 建议顺序：好二妹 → 好妹 → 灵儿

---

## 一、升级概述

v0.3.0 是 hermes-next 自发布以来最大的一次版本升级，核心目标是将已实现但未启用的认知管道接入真实生产流程，补齐 L3 持久化、生命周期管理等关键缺失。

| 维度 | v0.2.1（当前） | v0.3.0（目标） |
|------|---------------|---------------|
| 检索管道 | 内联简化版三路检索 | 6 步融合检索：语义 + FTS5 + 时间线 + RRF 融合 + Recency 提升 + MMR 去重 |
| 认知管道 | CognitivePipeline 代码已写但 provider.py 从未调用 | sync_turn 喂入 L1 traces → on_session_end 触发完整晋升链 |
| L1 Capture | 仅写 OpenViking | OpenViking + 本地 SQLite 双写，FTS5 全文搜索生效 |
| L2 Policy Induction | 可独立调用但未接入流程 | 会话结束时自动运行，结果持久化到 SQLite |
| L3 World Model | 纯内存，进程重启数据丢失 | Concepts + Triples 写入 SQLite，可选持久化到 OpenViking |
| Skill Crystallization | 可独立调用但未接入流程 | 会话结束时自动运行（默认关闭，需 opt-in） |
| 生命周期管理 | 无 | Trace 90 天归档、Policy 置信度 3%/天衰减、低分剪枝 |
| 监控工具 | 3 个工具（search/get/timeline） | 4 个工具（新增 memos_status 查看管道健康 + 晋升统计） |
| Viewer | Concepts/Triples 返回空数据 | Pipeline⚡ 面板 + 真实 Concepts/Triples 可视化 |
| 配置系统 | 仅检索配置 | 新增 cognitive + lifecycle 两大配置块 |

---

## 二、变更文件清单

```
 12 files changed, 998 insertions(+), 24 deletions(-)

 新增文件：
   hermes_next/cache/concepts.py     — Concept/Triple Repository（L3 持久化）
   hermes_next/cache/lifecycle.py    — LifecycleManager（归档/衰减/剪枝）

 修改文件：
   hermes_next/__init__.py           — 版本号 0.2.1 → 0.3.0
   hermes_next/cache/__init__.py     — 导出新 repository
   hermes_next/cache/schema.py       — 新增 concepts + triples 表
   hermes_next/config.py             — 新增 cognitive + lifecycle 配置
   hermes_next/provider.py           — 核心变更（详见第三节）
   hermes_next/viewer/server.py      — Pipeline 面板 + Concepts/Triples 视图
   plugin.yaml                       — 版本号同步
   pyproject.toml                    — 版本号同步
   tests/test_integration.py         — 12 个新集成测试
   tests/test_provider.py            — 适配 4 个工具断言
```

---

## 三、核心架构变更

### 3.1 provider 初始化流程（v0.3.0）

```
provider.initialize()
├── OpenVikingClient           ← 连接 OV 服务
├── CacheConnection + Schema   ← 初始化 SQLite（含新表）
├── RetrievalPipeline          ← 6 步融合检索
├── LifecycleManager           ← 新增：生命周期管理
├── CognitivePipeline          ← 新增：认知管道（按配置启用阶段）
│   ├── L1_CAPTURE（始终启用）
│   ├── REWARD（默认启用）
│   ├── L2_INDUCTION（默认启用）
│   ├── L3_WORLD_MODEL（默认关闭）
│   └── SKILL_CRYSTALLIZATION（默认关闭）
└── OVSession                  ← 会话管理
```

### 3.2 单次交互流程

```
用户输入 → prefetch(query)
              └── RetrievalPipeline.retrieve()
                  ├── Step 1: 语义搜索（OpenViking）
                  ├── Step 2: 全文搜索（SQLite FTS5）
                  ├── Step 3: 时间线上下文
                  ├── Step 4: RRF 融合排序
                  ├── Step 5: Recency 热度提升
                  └── Step 6: MMR 多样化去重

AI 响应 → sync_turn(user, assistant)
              ├── capture_trace() → OpenViking
              ├── TraceRepository.insert() → SQLite 缓存
              ├── LifecycleManager.on_trace_inserted()  ← 新增
              └── CognitivePipeline.process_trace()     ← 新增
```

### 3.3 会话结束流程

```
on_session_end()
├── CognitivePipeline.process_session_end()
│   ├── REWARD: 奖励反向传播（temporal discount=0.85）
│   ├── L2_INDUCTION: 跨 trace 策略归纳
│   ├── L3_WORLD_MODEL: 概念聚类 + 三元组提取
│   └── SKILL_CRYSTALLIZATION: 技能结晶化
├── _persist_pipeline_results()
│   ├── PolicyRepository.insert() → SQLite
│   ├── SkillRepository.insert() → SQLite
│   ├── ConceptRepository.insert() → SQLite  ← 新增
│   └── TripleRepository.insert() → SQLite   ← 新增
├── OVSession.commit()
└── 清空 session_traces
```

---

## 四、安装步骤

### 4.1 标准安装

```bash
pip install hermes-next==0.3.0 --force-reinstall --no-deps
```

### 4.2 源码安装

```bash
cd /d/WorkBuddy/workspace/hermes-next
pip install -e . --force-reinstall --no-deps
```

### 4.3 验证

```bash
python -c "from hermes_next import __version__; print(__version__)"
# 应输出: 0.3.0
```

---

## 五、配置说明

### 默认配置（直接可用）

```yaml
cognitive:
  auto_reward_on_session_end: true
  enable_l2_induction: true
  enable_l3_world_model: false         # GPT 密集型，默认关闭
  enable_skill_crystallization: false  # 默认关闭
  min_traces_before_l2: 5

lifecycle:
  trace_retention_days: 90
  policy_decay_rate: 0.03
  policy_min_confidence: 0.05
  cleanup_interval_traces: 500
```

---

## 六、注意事项

### 升级顺序
好二妹（先行）→ 观察 1-2 天 → 好妹 + 灵儿

### 数据兼容
- 自动创建 concepts + triples 新表
- 已有数据完全不受影响，无需迁移

### 回滚
```bash
pip install hermes-next==0.2.0 --force-reinstall --no-deps
```

### 行为变化
| 变化 | 影响 |
|------|------|
| sync_turn 写本地 SQLite | FTS5 搜索可用，cache.db 会增长但 90 天自动清理 |
| 会话结束触发认知管道 | 首次可能多 1-2 秒延迟 |
| Policy 置信度每日衰减 | 不用自动降级，无需手动清理 |
| memos_status 新工具 | Agent 可主动查询管道健康 |

---

## 七、验证清单

- [ ] `pip show hermes-next` 显示 0.3.0
- [ ] `__version__` 输出 `0.3.0`
- [ ] Provider 初始化无报错
- [ ] Viewer 侧边栏出现 Pipeline⚡、Concepts、Triples
- [ ] `memos_status` 返回管道信息
- [ ] 完成一次对话后 cache.db 的 `policies` 表有数据

---

## 八、技术概要

| 指标 | 值 |
|------|-----|
| 版本 | v0.3.0 |
| commit | d0591ff |
| tag | v0.3.0 |
| 文件变更 | 12 files, +998/-24 |
| 测试 | 118 passed |
| 依赖 | openviking>=0.3.22,<0.4 |
