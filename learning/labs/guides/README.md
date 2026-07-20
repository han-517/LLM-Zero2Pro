# Starter 与练习指南

这里说明如何完成 `learning/labs/starter/`，不是参考答案目录。课程包含 20 个文本主线练习和 1 个多模态选修；稳定编号不按文件名推断课次，以 `course/exercises.yaml` 为准。

```text
uv run llm-course exercises list
uv run llm-course exercises check 11
uv run llm-course exercises check autograd
```

## 目录职责

- `../starter/`：学习者需要填写的核心代码。
- `hints.md`：分层提示。
- `01_foundations.md`、`03_transformer.md`、`07_moe.md`：部分阶段练习说明。
- `checks/exercises/`：仓库根目录下的公开形状、边界、梯度和数值 oracle。
- `solutions/` 与 `src/llm_from_scratch/`：核查通过后再比较的答案和参考实现。

## 每题学习循环

1. 从 `learning/README.md` 完成本课讲义、互动图和实验。
2. 只打开当前 starter，先写出输入、输出和中间形状。
3. 构造最小反例，再填写核心空缺。
4. 运行单题核查，从第一条失败信息开始修正。
5. 公开核查通过后补自己的边界测试并口述复杂度。
6. 最后再查看参考实现。

`course check` 检查仓库课程闭环；`exercises check` 检查学员填写内容，两者互不替代。