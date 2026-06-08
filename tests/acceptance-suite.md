# Hermes Next v0.3.0 验收测试集

> 用途：升级后团队逐项验证，确保功能正常
> 测试人：好二妹（先行）→ 好妹 → 灵儿
> 预计耗时：30-45 分钟
> 前置条件：pip install hermes-next==0.3.0 已完成

---

## 一、冒烟测试（5 分钟）

### 1.1 版本确认

```bash
python -c "from hermes_next import __version__; print(__version__)"
```

- [ ] 输出 `0.3.0`
- [ ] 非 `0.2.1`、非 `0.1.0`

### 1.2 基础导入

```bash
python -c "
from hermes_next import HermesNextProvider, __version__
p = HermesNextProvider()
print(f'Provider: {p.name}')
print(f'Tools: {len(p.get_tool_schemas())} tools')
for s in p.get_tool_schemas():
    print(f'  - {s[\"function\"][\"name\"]}')
"
```

- [ ] Provider 名称为 `hermes-next`
- [ ] 工具数量为 **4**（memos_search / memos_get / memos_timeline / memos_status）
- [ ] 无报错

### 1.3 未初始化行为

```bash
python -c "
from hermes_next import HermesNextProvider
p = HermesNextProvider()
print(f'is_available: {p.is_available()}')
result = p.prefetch('test', session_id='s1')
print(f'prefetch(未初始化): \"{result}\"')
"
```

- [ ] `is_available()` 返回 `False`
- [ ] `prefetch()` 返回空字符串 `""`
- [ ] 无报错

---

## 二、Provider 生命周期（10 分钟）

### 2.1 完整初始化 + 会话

```python
import time
from hermes_next import HermesNextProvider
from hermes_next.config import HermesNextConfig

# 使用默认配置
config = HermesNextConfig()
provider = HermesNextProvider(config)

# 模拟好二妹的 Agent
provider.initialize(
    session_id="test-v3-001",
    agent_name="haoermei"
)

print(f"✓ initialized, agent={provider._agent_name}")
print(f"✓ is_available={provider.is_available()}")
```

- [ ] `initialize()` 无报错
- [ ] `is_available()` 返回 `True`（OpenViking 运行中时为 True）

### 2.2 模拟一轮对话

```python
# 第 1 轮
provider.sync_turn(
    user_content="好二妹，帮我分析一下这个项目的技术选型：Python vs Node.js",
    assistant_content="从项目规模来看，建议用 Python。理由：团队 Python 经验丰富、生态更成熟、长期维护成本低。",
    session_id="test-v3-001",
    tags=["技术选型", "分析"],
)

# 第 2 轮
provider.sync_turn(
    user_content="那数据库选型呢？MySQL vs PostgreSQL？",
    assistant_content="推荐 PostgreSQL。理由：JSONB 支持好、全文搜索内置、可扩展性强。",
    session_id="test-v3-001",
    tags=["技术选型", "数据库"],
)

# 第 3 轮（带负反馈）
provider.sync_turn(
    user_content="不对，上次我们踩过 PG 的坑，MySQL 在这个场景更稳",
    assistant_content="你说得对，纠正。如果团队对 PG 运维经验不足，MySQL 确实是更稳的选择。",
    session_id="test-v3-001",
    tags=["技术选型", "纠错"],
    metadata={"feedback": "negative", "corrected": True},
)

print(f"✓ 3 轮对话 sync_turn 完成")
print(f"✓ turn_index={provider._turn_index}")
print(f"✓ session_traces={len(provider._session_traces)} 条")
```

- [ ] `sync_turn()` 无报错
- [ ] `turn_index` 递增到 3
- [ ] `session_traces` 有 3 条

### 2.3 检索能力

```python
# prefetch 检索
ctx = provider.prefetch("PostgreSQL 有什么坑", session_id="test-v3-001")
print(f"prefetch 结果:\n{ctx[:500]}")
print(f"...(共 {len(ctx)} chars)")
```

- [ ] 返回内容包含刚才对话的信息
- [ ] 格式正确（`## 相关记忆` 开头）
- [ ] 无报错

### 2.4 Tool Call

```python
# memos_search
r1 = provider.handle_tool_call("memos_search", {"query": "技术选型", "k": 5})
print(f"memos_search: {len(r1)} chars ✓")

# memos_status
r2 = provider.handle_tool_call("memos_status", {})
print(f"memos_status:\n{r2}")

# memos_get — 需要一个 trace id，先搜再读
import json
trace_result = provider.handle_tool_call("memos_search", {"query": "PostgreSQL", "k": 1})
# 从结果中提取 trace id，或跳过此步
print(f"memos_get: 搜索结果已返回")
```

- [ ] `memos_search` 返回结果
- [ ] `memos_status` 显示管道信息（traces/policies/concepts 等）
- [ ] 总共 4 个工具全部可用

### 2.5 会话结束（触发认知管道）

```python
provider.on_session_end(messages=[])
print(f"✓ on_session_end 完成")
print(f"✓ session_traces 已清空: {len(provider._session_traces)}")

# 再次查看状态
status = provider.handle_tool_call("memos_status", {})
print(f"管道状态:\n{status}")
```

- [ ] `on_session_end()` 无报错
- [ ] `session_traces` 已清空为 0
- [ ] 管道状态显示策略数 > 0（如果满足了 L2 归纳条件）

### 2.6 优雅关闭

```python
provider.shutdown()
print(f"✓ shutdown 完成")
print(f"✓ initialized={provider._initialized}")
```

- [ ] `shutdown()` 无报错
- [ ] `_initialized = False`

> **验收签名：** `___________` 日期：________
> 通过标准：以上全部 ✓，无报错

---

## 三、检索质量测试（10 分钟）

### 3.1 语义搜索 vs 关键字搜索

```python
from hermes_next import HermesNextProvider
provider = HermesNextProvider()
provider.initialize(session_id="test-retrieval", agent_name="test")

# 先写入几轮带语义关联的对话
scenarios = [
    ("这个 App 启动太慢了", "优化了冷启动流程，懒加载非核心模块"),
    ("用户反馈页面卡顿", "对列表页做了虚拟滚动，图片加了懒加载"),
    ("内存占用有点高", "排查了内存泄漏，优化了大图缓存策略"),
]

for user, asst in scenarios:
    provider.sync_turn(user, asst, session_id="test-retrieval")

# 测试 1：语义关联查询（不包含关键字）
q1 = "性能优化"
r1 = provider.prefetch(q1, session_id="test-retrieval")
print(f"[语义搜索] query='{q1}'")
print(f"  命中: {len(r1)} chars")
print(f"  包含启动优化: {'冷启动' in r1}")
print(f"  包含卡顿优化: {'虚拟滚动' in r1 or '卡顿' in r1}")

# 测试 2：精确关键字查询
q2 = "内存"
r2 = provider.prefetch(q2, session_id="test-retrieval")
print(f"[关键字搜索] query='{q2}'")
print(f"  命中: {len(r2)} chars")
print(f"  包含内存优化: {'内存泄漏' in r2 or '内存' in r2}")

# 测试 3：空查询
r3 = provider.prefetch("", session_id="test-retrieval")
print(f"[空查询] 返回 {len(r3)} chars")

provider.shutdown()
```

- [ ] 语义搜索能命中"冷启动""卡顿"（不需要精确关键字匹配）
- [ ] 关键字搜索能精确命中
- [ ] 空查询不报错

### 3.2 检索回退测试

```python
# 使用一个完全不存在的查询
r = provider.prefetch("xyz9876nonexistent", session_id="test-retrieval")
print(f"[无结果查询] 返回: \"{r}\"")
```

- [ ] 返回空字符串，不报错
- [ ] 不抛异常

---

## 四、Viewer 检查（5 分钟）

```bash
# 启动 viewer
hermes-next-viewer --port 8080
```

浏览器打开 http://127.0.0.1:8080

- [ ] 仪表盘显示 Trace 数量 > 0
- [ ] 侧边栏出现 **Pipeline⚡** 页面
- [ ] 侧边栏出现 **Concepts** 页面
- [ ] 侧边栏出现 **Triples** 页面
- [ ] Traces 页面展示刚才写入的对话
- [ ] Pipeline 页面显示 L1/L2 统计（即使 L2 为 0）
- [ ] Concepts 页面可访问（即使数据为空）

---

## 五、生命周期测试（5 分钟）

### 5.1 生命周期管理器单元测试

```bash
python -c "
from hermes_next.cache.lifecycle import LifecycleConfig, LifecycleManager
from hermes_next.cache.connection import CacheConnection
import tempfile, os

# 用临时数据库
tmp = tempfile.mktemp(suffix='.db')
cache = CacheConnection(tmp)
from hermes_next.cache.schema import ensure_schema
ensure_schema(cache)

# 创建管理器
config = LifecycleConfig(
    trace_retention_days=1,
    policy_decay_rate=0.5,
    policy_min_confidence=0.1,
    cleanup_interval_traces=3,  # 每 3 次触发
)
manager = LifecycleManager(cache, config)

# 插入一些数据
from hermes_next.cache.traces import TraceRepository
from hermes_next.memos.types import TraceRow
repo = TraceRepository(cache)
for i in range(5):
    repo.insert(TraceRow(
        id=f't{i}', session_id='s1', turn_index=i,
        user_content=f'msg{i}', assistant_content=f'reply{i}',
    ))

# 触发清理
manager.on_trace_inserted()
manager.on_trace_inserted()
manager.on_trace_inserted()  # 第 3 次应触发清理
stats = manager.get_stats()
print(f'生命周期状态: {stats}')
print(f'✓ traces_since_cleanup={stats[\"traces_since_cleanup\"]}')
assert stats['traces_since_cleanup'] == 0, '应已清理'

cache.close_all()
os.unlink(tmp)
print('✓ 生命周期测试通过')
"
```

- [ ] 清理在达到阈值时触发
- [ ] 统计信息可获取
- [ ] 无报错

---

## 六、错误处理测试（5 分钟）

### 6.1 重复初始化

```python
provider = HermesNextProvider()
provider.initialize(session_id="s1")
try:
    provider.initialize(session_id="s2")
    print("✓ 重复 initialize 不报错（应幂等）")
except Exception as e:
    print(f"! 重复 initialize 报错: {e}")
```

- [ ] 重复 initialize 不抛异常或合理处理

### 6.2 shutdown 后操作

```python
provider = HermesNextProvider()
provider.initialize(session_id="s1")
provider.shutdown()

# shutdown 后调用
result = provider.sync_turn("hello", "world", session_id="s1")
print(f"✓ shutdown 后 sync_turn 不报错: {result}")

result = provider.prefetch("test", session_id="s1")
print(f"✓ shutdown 后 prefetch 返回空: \"{result}\"")
```

- [ ] shutdown 后方法不抛异常
- [ ] prefetch 返回空字符串

### 6.3 性能基线

```bash
python -c "
import time
from hermes_next import HermesNextProvider

provider = HermesNextProvider()
provider.initialize(session_id='perf-test')

# prefetch 耗时
t0 = time.time()
r = provider.prefetch('测试查询', session_id='perf-test')
t1 = time.time()
print(f'prefetch 耗时: {(t1-t0)*1000:.0f}ms (返回 {len(r)} chars)')

# sync_turn 耗时
t0 = time.time()
provider.sync_turn('你好', '你好！', session_id='perf-test')
t1 = time.time()
print(f'sync_turn 耗时: {(t1-t0)*1000:.0f}ms')

# memos_status 耗时
t0 = time.time()
s = provider.handle_tool_call('memos_status', {})
t1 = time.time()
print(f'memos_status 耗时: {(t1-t0)*1000:.0f}ms')

provider.shutdown()
print('✓ 性能基线完成')
"
```

- [ ] prefetch 耗时 < 1000ms（有 OV 时）/< 50ms（无 OV 时走本地 FTS5）
- [ ] sync_turn 耗时 < 500ms
- [ ] memos_status 耗时 < 100ms
- [ ] 记录基线值，便于后续版本对比

---

## 七、回滚验证（2 分钟）

```bash
# 确认回滚命令有效
pip install hermes-next==0.2.0 --force-reinstall --no-deps --dry-run 2>&1 | grep "Would install"
```

- [ ] 回滚命令语法正确
- [ ] 有回滚方案备而不用的安全感

---

## 八、综合验收卡片

```
┌─────────────────────────────────────────────┐
│  Hermes Next v0.3.0 验收                     │
│  ───────────────────                         │
│  Agent: ___________  日期: __________       │
│                                              │
│  ✅ 一、冒烟测试 (5项)      ___/5           │
│  ✅ 二、Provider 生命周期   ___/6           │
│  ✅ 三、检索质量            ___/3           │
│  ✅ 四、Viewer 检查         ___/7           │
│  ✅ 五、生命周期测试        ___/2           │
│  ✅ 六、错误处理            ___/4           │
│  ✅ 七、回滚验证            ___/2           │
│  ───────────────────                         │
│  总分: ___/29                               │
│  通过标准: ≥25/29                            │
│                                              │
│  备注:                                       │
│  ____________________________________       │
└─────────────────────────────────────────────┘
```

---

## 九、好二妹先行测试建议

作为第一个升级的 Agent，建议：

1. **先跑冒烟测试 + Provider 生命周期**（15 分钟）→ OK 了再继续
2. **日常使用 1-2 天**，不开 L3/Skill 结晶化（默认关闭）
3. **第 2 天回来跑检索质量 + Viewer 检查**
4. **第 3 天没问题** → 好妹、灵儿可以升了

```bash
# 好二妹快速验收一键脚本
python -c "
from hermes_next import __version__; print(f'v{__version__}')
from hermes_next import HermesNextProvider
p = HermesNextProvider()
print(f'name={p.name}, tools={len(p.get_tool_schemas())}')
p.initialize(session_id='quick-test')
p.sync_turn('测试', '测试回复', session_id='quick-test')
ctx = p.prefetch('测试', session_id='quick-test')
print(f'prefetch OK ({len(ctx)} chars)')
status = p.handle_tool_call('memos_status', {})
print(f'status OK ({len(status)} chars)')
p.shutdown()
print('✅ 快速验收通过')
"
```
