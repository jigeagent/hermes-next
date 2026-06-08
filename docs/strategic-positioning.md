# Hermes Next × cc-star 战略定位分析

> 背景：memos-local-plugin v2.0（MemTensor/MemOS, 9411 stars, Apache-2.0）
> 已发布到 @memtensor/memos-local-openclaw-plugin v1.0.10（NPM）
> 我们拥有：hermes-next（多Agent记忆底座）+ cc-star（Claude Code 记忆升级包）

---

## 一、竞争对手分析：memos-local-plugin v2.0

### 他们是什么
MemTensor 团队维护的 TypeScript 记忆系统，Hermes Agent 和 OpenClaw 的官方记忆插件。

### 他们强在哪
- **社区认可度高**：9411 stars，Apache-2.0 开源
- **先发优势**：最早的 LLM Agent 记忆插件之一，生态认知 = memos
- **功能完整**：L1/L2/L3/Skill 四层都有，且在生产跑过
- **文档完善**：SegmentFault 有中文深度文章，GitHub 有 ARCHITECTURE.md

### 他们弱在哪
- **JSON-RPC 桥接**：TypeScript core → Python Hermes Agent 走 stdio 桥，偶发断连
- **检索方式**：3-Tier（Skill→Trace→WorldModel）+ 余弦暴力搜索，FTS5 都没有
- **无生命周期**：traces/policies 永远累积，DB 只增不删
- **无原生记忆打通**：跟 Hermes Agent 的 MEMORY.md/USER.md 互不感知
- **社区但不开阔**：9411 stars 但文档只有中文，国际社区渗透弱

---

## 二、我们的差异化定位

### hermes-next 的核心差异

| 维度 | memos-local v2.0 | hermes-next v0.3.2 |
|------|------------------|-------------------|
| 语言 | TypeScript + JSON-RPC | Python 原生（同进程） |
| 检索 | 3-Tier 余弦暴力 | 6步融合（语义+FTS5+RRF+MMR+Recency） |
| 生命周期 | 无 | ✓ 90天归档+置信度衰减+剪枝 |
| 原生记忆打通 | ❌ | ✓ MEMORY.md 晋升+session_search回退 |
| 跨 Agent 共享 | JSON-RPC | ✓ OpenViking 命名空间 |
| 配置热更新 | ❌ 需重启 | ✓ reinitialize 即生效 |

### cc-star 的核心差异
- Claude Code 生态唯一类脑记忆升级包
- 桥接 cache.db ↔ MEMORY.md 的双向晋升
- "不是替代原生记忆，是做增强"的路线已被验证

### 一句话定位
> **memos-local 是 "够用的记忆插件"，hermes-next 是 "会进化的数字大脑"**

---

## 三、"打透"意味着什么

### 3.1 产品层面

**hermes-next 还需要补齐的短板（对标 memos v2.0 旗舰功能）：**

| 功能 | 优先级 | 参考 memos 做法 |
|------|--------|----------------|
| 启动恢复（进程重启不丢数据） | P0 | `init()` 扫描孤儿 episodes |
| Feedback 体验闭环 | P0 | `submitFeedback` → L2/L3 重跑 |
| Decision Repair（决策修复） | P1 | `@repair` 块写入 Policy |
| SSE 实时 Viewer | P2 | `subscribeEvents` 实时面板 |
| LLM 检索精排 | P2 | `llmFilterCandidates` |
| Telemetry 监控 | P3 | ARMS 集成 |

**cc-star 还需要：**
| 功能 | 优先级 |
|------|--------|
| MEMORY.md 双向同步（cc-star→原生←cc-star） | P1 |
| 多项目配置文件切换 | P2 |
| PyPI 稳定发布 + GitHub CI 全绿 | P0 |

### 3.2 社区层面

"打透" = 两个社区 + 国际频道：

```
中文社区                 国际社区
┌────────────┐          ┌────────────┐
│ 知乎/掘金   │          │ Reddit     │
│ SegmentFault│          │ HackerNews │
│ 公众号      │          │ Discord    │
│ 飞书开放平台│          │ Twitter/X  │
└────────────┘          └────────────┘
         ↓                     ↓
   核心叙事统一：                                   
   "memos 是 TypeScript 时代的产物，
    hermes-next 是 Python 原生时代的答案"
```

**内容策略建议：**
1. **第一篇**（中文）："从 memos-local 到 hermes-next——为什么我们放弃了 JSON-RPC 桥接"
2. **第二篇**（英文）："Hermes Next: A Python-native memory system that learns like a brain"
3. **第三篇**（中英同步）："cc-star: Upgrading Claude Code memory without replacing it"
4. **持续输出**：GitHub Release Notes + 架构决策记录（ADR）

---

## 四、路线图建议

```
Phase 1（现在）         Phase 2（7月）            Phase 3（8月）
┌─────────────┐       ┌──────────────┐          ┌──────────────┐
│ hermes-next  │       │ hermes-next   │          │ 社区发声      │
│ v0.3.2 稳定  │──────→│ v0.4.0        │─────────→│              │
│ cc-star     │       │ ✅ Feedback    │          │ 知乎首篇     │
│ v0.3 打磨    │       │   体验闭环     │          │ Reddit 同步  │
│ 好二妹验证中 │       │ ✅ 启动恢复    │          │ GitHub 开源  │
│             │       │ ✅ Decision    │          │ Discord 运营 │
│             │       │    Repair      │          │              │
│             │       │ cc-star v0.4   │          │              │
│             │       │ ✅ 双向同步     │          │              │
└─────────────┘       └──────────────┘          └──────────────┘
```

---

## 五、我的判断

**这个方向完全值得干，且时机正好。** 理由是：

1. **memos 有 9411 颗 star 但没出圈** → 说明需求真实存在，但市场没有被满足好
2. **我们有两个产品的实战验证** → hermes-next 在好妹/好二妹/灵儿生产跑，cc-star 在 Claude Code 生态跑
3. **差异化足够清晰** → Python 原生 vs TypeScript + 桥接，这是根本性的架构优势
4. **开源即获客** → 把 hermes-next 放到 GitHub 上，memos 的用户自然会被吸引过来比较

建议下一步：**先把 hermes-next 放到 GitHub 上开源**（至少 public 可读），然后写那篇对比文章。9425 个 star 的用户看到"Python 原生替代 JSON-RPC 桥接"，至少会点进来看一眼。🤛
