# Hot Context — 跨会话续接

> 让好妹/好灵儿用 hermes 时，新会话自动续接上次工作进度。

---

## 一句话

每次对话结束时自动记下摘要，下次启动时注入 system prompt，不用重复说"上次做到哪了"。

## 启用

不需要改配置，hermes-next v0.7 自带开启。验证是否生效：

新会话启动后，对好灵儿说：

> 看看 system prompt 里有 "Previous session snapshot" 吗？

有就是生效了。

## 效果

```
没 hot.md 时：
  好妹：刚才那个方案讨论到哪了？
  Agent：什么方案？（失忆.jpg）

有 hot.md 时：
  好妹：刚才那个方案讨论到哪了？
  Agent：上次到 Phase 4 方案评审，康少提了 3 个风险点……
```

## 文件位置

`~/.hermes/hot.md`，纯文本，可以直接打开看和改。

- 想重置 → 清空文件内容
- 想纠正 → 直接编辑里面的摘要
- 想查看 → `cat ~/.hermes/hot.md`
