# 教程讲义入口

这里保存学习者真正需要阅读的课程正文。进入本目录后，请先选择路线，不要从文件名随机打开。

- [15 周核心路线](core_learning_path.md)：第一次系统学习或希望先建立主干。
- [48 周完整路线](full_learning_path.md)：从基础、预训练到后训练与推理服务全部完成。
- [逐周讲义目录](weeks/README.md)：讲义写作结构与质量契约。
- [互动实验入口](interactive/index.html)：在阅读公式后改变参数、观察反例。
- [多模态选修](extensions/multimodal.md)：48 周文本主线之外的扩展。

## 三层内容各负责什么

| 层 | 目录 | 用法 |
|---|---|---|
| 逐周正文 | `docs/weeks/NN_*.md` | 当前周的主要阅读材料；包含推导、代码、反例、实验和验收 |
| 阶段综述 | `docs/stages/*.md` | 回顾一个阶段的知识地图；不是完整周讲义 |
| 可运行实验 | `notebooks/core/*.ipynb` | 验证讲义中的最小机制；同一本 Notebook 可能服务多周 |

课程配置位于 `course/stages/*.yaml`，它只负责把“周次 → 讲义 → Notebook → starter →
来源 → 交付物”连接起来。学习者通常不需要读 YAML。

## 正确的每周顺序

1. 从路线表找到当前“学习单元/原课程周”。
2. 点击“本周讲义”，先写下直觉、shape 和公式预测。
3. 打开同一行 Notebook，只完成当前周相关单元。
4. 填写路线表列出的 starter，或按 deliverable 产生研究记录。
5. 运行 `uv run llm-course exercises check <ID>`。
6. Restart Kernel and Run All Cells，再按讲义 rubric 验收。
7. 在 `progress.yaml` 更新原课程周状态。

## 为什么不再让多周共用一篇短文

旧结构让 3–7 个教学周反复指向同一篇阶段摘要，平均每周正文不足约一千字符，容易把课程提纲
误认为完整讲义。现在 48 周必须各自指向唯一的 `docs/weeks/NN_*.md`，并由
`uv run llm-course course check --no-tests` 自动验证正文长度、必要结构、代码块、一手来源和
本地链接。阶段综述仍保留，但只用于导航和复盘。

## 重新生成路线表

路线表来自课程清单，不应手工维护：

~~~text
uv run llm-course course path --weeks 15 --write
uv run llm-course course path --weeks 48 --write
~~~

修改任一周 lecture、Notebook 或 starter 映射后，需要重新运行以上命令并执行课程健康检查。
