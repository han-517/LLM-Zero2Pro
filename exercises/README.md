# 练习区

这里不是参考答案目录，而是“学完后亲手补核心实现”的工作区。课程包含 20 个文本 LLM 主线练习和 1 个多模态选修；编号 01–10 为兼容保留，展示顺序按真实周次排列。

## 从哪里开始

```text
uv run llm-course exercises list
uv run llm-course exercises check 11
uv run llm-course exercises check autograd
```

`exercises/manifest.yaml` 是编号、别名、周次、starter 与 checker 的唯一数据源。不要根据文件名猜周次。

## 目录职责

- `starter/`：需要填写的核心代码；初始 `NotImplementedError` 是正常状态。
- `checks/`：公开的形状、边界、梯度和小规模数值 oracle，不泄露完整答案。
- `hints.md`：分层提示。
- `solutions/`：精选解法与验收思路，最后再看。
- `manifest.yaml`：机器可读练习契约。

## 每题学习循环

1. 完成 roadmap 对应讲义、互动图和 Notebook。
2. 只打开当前 starter，写出输入/输出形状。
3. 运行单题核查，从第一条失败信息构造最小反例。
4. 公开核查通过后再补一个自己的边界测试。
5. 解释复杂度、数值稳定性和一个常见错误。
6. 最后再与 `src/llm_from_scratch/` 参考实现比较。

`uv run llm-course course check` 检查仓库和 48 周资产闭环；`uv run llm-course exercises check ...` 才检查学员填写内容。两类命令严格分离。

