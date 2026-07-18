# Starter 练习

这些文件故意保留 TODO，不参加仓库默认的 `pytest`。请先独立完成，再阅读 `src/llm_from_scratch/` 参考实现。

运行方式：

```powershell
uv run python exercises/starter/01_stable_softmax.py
uv run python exercises/starter/02_causal_attention.py
uv run python exercises/starter/03_kv_cache_budget.py
uv run python exercises/starter/04_sft_shift.py
uv run python exercises/starter/05_moe_capacity.py

# 推荐：统一列出并核查 10 个模板
uv run llm-course exercises list
uv run llm-course exercises check 01
uv run llm-course exercises check rope
uv run llm-course exercises check all
# 详细说明: docs/code_templates.md
```

建议顺序：

1. 先写一个必然失败的最小反例。
2. 写出输入、输出形状。
3. 完成 TODO。
4. 让文件末尾断言通过。
5. 再补一个你自己想到的边界输入。
