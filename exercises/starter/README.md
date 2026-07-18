# Starter 练习

这些文件故意保留 `TODO` 和 `NotImplementedError`，不参加仓库默认 `pytest`。请先完成对应讲义、互动图和 Notebook，再在这里独立实现；公开核查通过后才查看 `src/llm_from_scratch/`。

```text
uv run llm-course exercises list
uv run llm-course exercises check 11
uv run llm-course exercises check autograd
uv run llm-course exercises check all
```

编号、别名、真实周次和 checker 均来自 `../manifest.yaml`。当前有 20 个文本主线模板和 1 个多模态选修，旧编号 01–10 保持兼容。

推荐顺序：

1. 写出输入、输出和中间张量形状。
2. 先构造一个必然失败的最小反例。
3. 只填写当前文件中的核心空缺。
4. 运行单题 checker，从第一条失败信息开始修正。
5. 再补一个非法输入、边界输入或有限差分检查。
6. 口述复杂度和教学实现的边界，最后才对照参考实现。

完整地图见 `../../docs/code_templates.md`。
