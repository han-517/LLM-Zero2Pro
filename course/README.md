# course：课程控制中心

这个目录不放长篇讲义，也不放学员代码。它负责回答三个问题：

1. 48 周怎样分阶段、每周学什么？
2. 每周对应哪篇讲义、哪个 Notebook、哪个 starter/checker？
3. 完成这一周的可核查产出是什么？

真正的知识讲解在 `docs/`，实验在 `notebooks/`，填空实现与核查在
`exercises/`。这里是把这些资产连接起来的“课程索引”，不是第四套教程。

## 目录结构

```text
course/
├── README.md                 # 你正在看的说明
├── roadmap.yaml              # 很小的入口清单，只列出下面文件
├── course.yaml               # 课程元数据、边界与 15 周核心路径
└── stages/
    ├── 01_foundations.yaml
    ├── 02_neural_lm.yaml
    ├── 03_transformer.yaml
    ├── 04_modern_decoder.yaml
    ├── 05_pretraining.yaml
    ├── 06_attention_frontiers.yaml
    ├── 07_moe.yaml
    ├── 08_posttraining.yaml
    └── 09_inference_capstone.yaml
```

每个阶段文件只管理自己的周次。每个 `lesson` 同时包含：

- 学习目标、阅读、实验、练习和验收；
- `assets.lecture`：本周讲义；
- `assets.notebooks`：本周实验；
- `assets.exercises`：starter/checker ID；
- `assets.sources`：一手来源；
- `assets.deliverable`：应提交或记录的产出。

把 lesson 与 assets 放在同一周下面，避免过去在一个 700 多行文件的前后两处来回查找。

## 学员怎么使用

通常不需要直接编辑 YAML。先生成适合阅读的路线：

```text
uv run llm-course course path --weeks 15
uv run llm-course course path --weeks 48
```

然后用统一入口启动：

```text
uv run llm-course lab
```

## 维护者怎么修改

1. 只修改对应的 `stages/NN_*.yaml`。
2. 若调整 15 周核心路线，再修改 `course.yaml` 的 `paths.core_15`。
3. 不在多个文件重复维护周次或 Notebook 路径。
4. 运行：

```text
uv run python scripts/sync_notebooks.py
uv run llm-course course path --weeks 15 --write
uv run llm-course course check --no-tests
```

加载器会把入口清单、课程信息和 9 个阶段聚合成兼容的内存结构，因此 CLI 和测试仍可按
`weeks`、`stages`、`assets` 查询，但磁盘结构保持清晰。
