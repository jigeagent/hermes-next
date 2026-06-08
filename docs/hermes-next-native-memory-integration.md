# Hermes Agent 原生记忆 × Hermes Next v0.3 融合方案

> 思路来源：cc-star v0.3 整合 Claude Code 原生记忆的成功模式
> 核心理念：**不替代，做增强。原生做导航，hermes-next 做大脑。**

---

## 一、从 cc-star 经验中继承的原则

cc-star v0.3 能跟 Claude Code 原生记忆和平共处的关键四条：

### 原则 1：MEMORY.md = 导航牌，不是记忆库
原生记忆文件**只存精华**。cc-star 实践证明，MEMORY.md 维持在 45-50% 使用率是健康状态——里面放的是**索引、关键事实、经验摘要**，不是原始对话。删掉一条会答错，才配留在 MEMORY.md。

### 原则 2：检索链四层，自动降级
```
Layer 0: MEMORY.md（热点）→ Layer 1: hermes-next（语义+FTS5）
→ Layer 2: session_search（关键字回退）→ Layer 3: OV（团队共享）
```
上一层无结果，自动 fallthrough 到下一层。Agent 不需要知道自己用的是什么。

### 原则 3：晋升管道自动化
```
原始数据 → 结构化记忆 → 归纳提炼 → MEMORY.md 精华
     ↑           ↑            ↑            ↑
  session   hermes-next   L2 Policy   自动写入
  raw text   capture      归纳成功     MEMORY.md
```

### 原则 4：原生能力永不破坏
即使 hermes-next 挂了，原生 MEMORY.md + session_search 照常工作。

---

## 二、4 层检索链设计

```
┌─────────────────────────────────────────────────────────┐
│ Layer 0: Hot Memory（始终在 system prompt）              │
│ ─────────────────────────────────────────────────────── │
│ MEMORY.md / USER.md                                      │
│ 容量：2,200 chars + 1,375 chars ≈ 1,300 tokens          │
│ 内容：导航牌——索引/关键事实/已验证经验/团队SOP引用       │
│ 写入方式：                                            │
│   ① 手动 memory add（紧急/重要）                         │
│   ② hermes-next L2→L3 晋升后自动写入摘要                │
│   ③ session_search 高命中条目自动推广                   │
├─────────────────────────────────────────────────────────┤
│                        ▼ fallthrough                     │
├─────────────────────────────────────────────────────────┤
│ Layer 1: Hermes Next（主检索通道，每轮自动 prefetch）    │
│ ─────────────────────────────────────────────────────── │
│ 覆盖 95% 的检索需求                                     │
│ 语义搜索（OpenViking）+ FTS5 全文 + Policy 匹配         │
│ + RRF 融合 + Recency 提升 + MMR 去重                    │
│ 耗时 50-300ms，后台 prefetch 不阻塞                     │
├─────────────────────────────────────────────────────────┤
│                        ▼ fallthrough                     │
├─────────────────────────────────────────────────────────┤
│ Layer 2: Session Search（原生回退，关键字精确匹配）      │
│ ─────────────────────────────────────────────────────── │
│ Hermes Agent 内置 SQLite FTS5                            │
│ 耗时 15-50ms                                            │
│ 当 Layer 1 返回 0 结果时自动触发                        │
│ 命中后 → 标记该 Entry → 下次晋升为 hermes-next trace    │
├─────────────────────────────────────────────────────────┤
│                        ▼ fallthrough                     │
├─────────────────────────────────────────────────────────┤
│ Layer 3: OpenViking（团队共享，跨 Agent）                │
│ ─────────────────────────────────────────────────────── │
│ 好妹/好二妹/灵儿共享记忆                                │
│ 命名空间隔离 + Hub 风格跨 Agent 搜索                    │
│ 耗时 100-500ms                                          │
└─────────────────────────────────────────────────────────┘
```

### 工作示例

```
用户提问："上次那个 RAG 优化方案的核心结论是什么？"

Step 1: MEMORY.md 扫描（在 prompt 中）
  → 命中"攀枝花→session_search("攀枝花")"索引
  → 没直接找到 RAG 相关内容，fallthrough

Step 2: hermes-next prefetch
  → 语义搜索 "RAG 优化方案 核心结论"
  → 命中 3 条相关 trace，score 0.85/0.72/0.61
  → 注入 prompt → 直接回答
  → ✅ 走通，结束

如果 Step 2 无结果：
Step 3: session_search("RAG 优化")
  → FTS5 精确匹配到历史会话
  → 返回文本片段
  → 标记该条目为"高价值"→ 后台触发 promotion
```

---

## 三、晋升管道：从原生记忆到 hermes-next 再到 MEMORY.md

### 3.1 晋升路径

```
                   ┌──────────────────────┐
                   │  MEMORY.md 精华       │  ← L2 Policy 摘要 / 关键参考
                   │  2,200 chars           │  ← session_search 高命中条目
                   └──────┬───────────────┘
                          │ 自动写入（promoter）
      ┌───────────────────┼───────────────────┐
      │                   │                   │
      ▼                   ▼                   ▼
┌────────────┐  ┌──────────────┐  ┌────────────────┐
│ L2 Policy  │  │ L3 Concept   │  │ Skill          │  ← hermes-next 认知管道输出
│ 经验模式   │  │ 环境认知      │  │ 可复用技能      │
└────────────┘  └──────────────┘  └────────────────┘
      ▲                ▲                 ▲
      │                │                 │
      └────────────────┼─────────────────┘
                       │ L2/L3/Skill 归纳
                       │
              ┌────────────────┐
              │ hermes-next    │  ← 核心认知引擎
              │ 原始 Traces    │  ← 每次对话自动 capture
              └────────────────┘
                       ▲
                       │ onTurnEnd / sync_turn
                       │
              ┌────────────────┐
              │ Session Traces │  ← 原始对话数据
              │（原生 SQLite）  │
              └────────────────┘
```

### 3.2 晋升触发器

| 触发条件 | 动作 | 目标 |
|---------|------|------|
| session_search 同一查询命中 >3 次 | 创建 Trace 写入 hermes-next | Layer 1 下次可命中 |
| L2 Policy 置信度 >0.5 | 摘要写入 MEMORY.md | Agent 下次启动即见 |
| L3 Concept 成员 trace >5 | 概念描述写入 MEMORY.md | 团队共享环境认知 |
| Skill 结晶化成功 | Skill 名称 + 用法写入 MEMORY.md | 团队可复用 |
| MEMORY.md 容量 >80% | 触发瘦身：低引用条目下沉到 hermes-next | 腾出空间 |

### 3.3 具体做法

在 hermes-next 的 `_persist_pipeline_results()` 成功写入新 policy/skill 后，增加回调：

```python
# provider.py (v0.3 + integration)
def _promote_to_memory_md(self, entry: str) -> None:
    """将高价值内容写入 Hermes Agent 原生的 MEMORY.md"""
    memory_path = Path.home() / ".hermes" / "memories" / "MEMORY.md"
    if not memory_path.exists():
        return
    # 用原生 memory tool 的格式追加
    # § 分隔，150 chars 以内
    summary = entry.strip()[:150]
    with open(memory_path, "a", encoding="utf-8") as f:
        f.write(f"\n{summary}\n§\n")
```

---

## 四、配置方案

```yaml
# hermes-next.yaml 新增融合配置
integration:
  native_memory:
    sync_memory_md: true                # 是否自动写入 MEMORY.md
    memory_md_capacity_warning: 0.8     # 超过 80% 触发瘦身
    promote_session_hits: 3             # session_search 命中几次提升
    promote_on_l2_confidence: 0.5       # L2 多高置信度提升至 MEMORY.md
    promote_on_skill_crystallize: true  # Skill 结晶化后写入概要
  
  retrieval_chain:
    order: ["hermes_next", "session_search", "openviking"]
    hermes_next_min_score: 0.4          # 低于此分 fallthrough
    session_search_fallback: true       # 是否启用 fallback
    mutual_promotion: true              # 双向提升机制
```

---

## 五、和 cc-star v0.3 的对照

cc-star v0.3 做的事 | Hermes Next 融合版对标 |
---|---|
MEMORY.md = 导航牌，cache.db = 记忆库 | MEMORY.md = 导航牌，hermes-next cache.db = 记忆库 |
四级检索链：Memory > session_search > cache.db > OV | 四层检索链：MEMORY.md > hermes-next > session_search > OV |
promote.py 自动将 cache.db 精华写入 MEMORY.md | `_promote_to_memory_md()` 将 L2/L3/Skill 写入原生 |
记忆瘦身三原则 | 同一套原则：留核去肉/同类合并/动态剥离 |
cc-star 故障 → Claude Code 原生记忆正常工作 | hermes-next 故障 → MEMORY.md + session_search 正常 |
双存储：SQLite + OV | 双存储：SQLite 原生 + hermes-next cache.db + OV |

---

## 六、实施路线图

### Phase 1（本周）
1. 在 hermes-next 实现 `_promote_to_memory_md()` 方法
2. L2 Policy 成功归纳后自动写入 MEMORY.md 摘要
3. 实现检索链 fallthrough 逻辑（Layer 1 无结果 → Layer 2）

### Phase 2（下周）
4. session_search 高命中条目自动 promotion
5. L3 Concept → MEMORY.md 自动摘要
6. MEMORY.md 容量监控 + 自动瘦身

### Phase 3（本月）
7. OpenViking 跨 Agent 共享层集成
8. 团队共享记忆的 Hub 风格搜索
9. 原生 Skills 与 hermes-next Skills 双向同步

---

## 七、总结

> **原生做 "稳" —— hermes-next 做 "强"**

| 场景 | 谁负责 | 备注 |
|------|--------|------|
| "你现在叫什么名字？" | MEMORY.md | 热点命中，0 token 消耗 |
| "上次 RAG 方案结论" | hermes-next | 语义检索，跨会话关联 |
| "查一下上周三的对话" | session_search | 精确时间范围，FTS5 最快 |
| "好二妹遇到过这个坑吗？" | OpenViking | 跨 Agent 共享 |
| 所有通道都挂了 | MEMORY.md | 降级到最基础，不失忆 |
