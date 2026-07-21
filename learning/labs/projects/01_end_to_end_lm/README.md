# 大作业 01：从字节 BPE 到可恢复训练的完整语言模型

目标不是再运行一次参考代码，而是在本目录维护一个完全属于你的 `student_lm`
包。项目代码不得导入仓库的参考实现；Notebook 可以用来画图，但最终训练必须从
`run_train.py` 的干净进程启动。

## 前置课次

完成第 9–21 课，并至少通过 starter 06、02、13、09、07、15。

## 固定顺序

1. **Tokenizer**：完成 byte vocabulary、确定性 merge、Unicode round-trip、特殊 token、保存/加载。
2. **Model**：完成 RMSNorm、SwiGLU、RoPE、因果注意力、Block 和 Transformer LM。
3. **Training**：完成 next-token batch、交叉熵、AdamW、梯度裁剪、训练循环和 checkpoint。
4. **Integration**：从 `data/tiny_corpus.txt` 训练，保存模型，重新启动进程后恢复并生成文本。
5. **Report**：填写 `report_template.md`，记录配置、参数量、token 数、loss、耗时和失败案例。

## 文件职责

```text
student_lm/tokenizer.py   # 你实现的 tokenizer
student_lm/model.py       # 你实现的 Transformer LM
student_lm/training.py    # 你实现的优化、训练与恢复
run_train.py              # 已给定的端到端入口，不导入参考实现
data/tiny_corpus.txt      # 离线小语料，只用于正确性与过拟合
report_template.md        # 实验报告
```

## 核查与运行

公开核查按 tokenizer → model → training 排列：

```text
uv run llm-course projects check 01
uv run python learning/labs/projects/01_end_to_end_lm/run_train.py --steps 120
```

公开测试只验证小规模行为，不替代最终验收。不要阅读 `src/llm_from_scratch/`
后照抄；先让失败测试缩小问题，再用手算张量验证。

## 完成标准

- 公开核查全部通过，项目包中不存在参考实现导入。
- 训练 loss 在固定 seed 的小语料上明显下降，所有梯度有限。
- checkpoint 恢复后模型、优化器、step 和随机数状态一致。
- 从新进程加载 checkpoint 能生成文本，不依赖 Notebook 隐藏状态。
- 报告给出数据哈希、配置、参数量、训练 token、loss 曲线、运行时间和至少两个失败案例。

本项目是 CPU correctness baseline；性能工程在大作业 03 中进行。
