# Starter 练习

这些文件故意保留 `TODO` 和 `NotImplementedError`，不参加仓库默认 `pytest`。先按 [`learning/README.md`](../../README.md) 完成当前课，再独立填写指定 starter。

```text
uv run llm-course exercises list
uv run llm-course exercises check 11
uv run llm-course exercises check autograd
```

编号、别名、真实课次和 checker 均来自 [`course/exercises.yaml`](../../../course/exercises.yaml)。当前有 20 个文本主线模板和 1 个多模态选修。

推荐顺序：

1. 写出输入、输出和中间张量形状。
2. 先构造一个必然失败的最小反例。
3. 只填写当前文件中的核心空缺。
4. 运行单题 checker，从第一条失败信息开始修正。
5. 再补非法输入、边界输入或有限差分检查。
6. 口述复杂度与教学实现边界，最后才对照 `src/llm_from_scratch/`。

完整地图见 [代码模板与核查](../../readings/references/code_templates.md)。