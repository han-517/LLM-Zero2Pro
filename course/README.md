# `course/`：课程索引

这里把统一的 48 个学习单元连接到 Notebook、练习和验收。它不是另一套教程。

## 结构

- `course.yaml`：课程元数据。
- `stages/`：每周的主题、资源和验收。
- `roadmap.yaml`：列出阶段文件。

## 学习者怎么用

直接打开[统一学习路线](../learning/README.md)。第一次从第 1 周开始，以后从第一个未完成的周继续。

## 维护者怎么用

修改 `stages/NN_*.yaml` 后，运行 `uv run llm-course course path --write` 更新唯一的 `learning/README.md`，再运行校验。

这里不再保存 15 周抽样路径，也不会生成两个版本。
