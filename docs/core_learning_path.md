<!-- 由 uv run llm-course course path --weeks 15 --write 生成，请勿手工维护表格。 -->
# 15 周核心路线

这份路径由 course/roadmap.yaml 聚合 course.yaml 与 stages/。每周都按讲义、CPU Notebook、
starter/研究产出、自动核查或 rubric、一手来源形成闭环。
互动实验统一入口：[交互式实验](interactive/index.html)；代码练习清单：
[exercises/manifest.yaml](../exercises/manifest.yaml)。二者都由课程健康检查纳入契约。

15 周路线从完整课程中抽取关键单元。学习单元按 1–15 连续进行；
“原课程周”保留 48 周路线的资产编号，所以出现跳号是正常的。

| 学习单元 | 原课程周 | 主题 | 讲义 | Notebook | Starter / 产出 |
|---:|---:|---|---|---|---|
| 1 | 1 | 命令行、Git、Python 与可复现环境 | [本周讲义](weeks/01_environment_reproducibility.md) | [00_START_HERE.ipynb](../notebooks/00_START_HERE.ipynb) | 环境诊断快照与复现命令 |
| 2 | 2 | 向量、矩阵与张量形状 | [本周讲义](weeks/02_tensor_shapes.md) | [01_shapes_and_autograd.ipynb](../notebooks/core/01_shapes_and_autograd.ipynb) | 三段张量代码的逐步形状表 |
| 3 | 3 | 导数、链式法则与自动微分 | [本周讲义](weeks/03_chain_rule_autograd.md) | [01_shapes_and_autograd.ipynb](../notebooks/core/01_shapes_and_autograd.ipynb) | 11 |
| 4 | 4 | 概率、Softmax、交叉熵与 PyTorch | [本周讲义](weeks/04_softmax_cross_entropy.md) | [01_shapes_and_autograd.ipynb](../notebooks/core/01_shapes_and_autograd.ipynb) | 01 |
| 5 | 7 | 词向量与神经语言模型 | [本周讲义](weeks/07_embeddings_neural_lm.md) | [02_neural_language_models.ipynb](../notebooks/core/02_neural_language_models.ipynb) | 12 |
| 6 | 10 | 从零实现 BPE | [本周讲义](weeks/10_byte_bpe_from_scratch.md) | [03_tokenization_and_bpe.ipynb](../notebooks/core/03_tokenization_and_bpe.ipynb) | 06 |
| 7 | 12 | 缩放点积注意力 | [本周讲义](weeks/12_scaled_dot_product_attention.md) | [04_attention_mechanics.ipynb](../notebooks/core/04_attention_mechanics.ipynb) | 02 |
| 8 | 13 | 因果掩码与多头注意力 | [本周讲义](weeks/13_causal_mask_and_multihead_attention.md) | [04_attention_mechanics.ipynb](../notebooks/core/04_attention_mechanics.ipynb) | 02 |
| 9 | 15 | 训练并采样 Tiny GPT | [本周讲义](weeks/15_train_and_sample_tiny_gpt.md) | [05_tiny_gpt.ipynb](../notebooks/core/05_tiny_gpt.ipynb) | 13 |
| 10 | 16 | Pre-Norm 与 RMSNorm | [本周讲义](weeks/16_rmsnorm_prenorm.md) | [06_modern_decoder.ipynb](../notebooks/core/06_modern_decoder.ipynb) | 09 |
| 11 | 18 | RoPE：用旋转编码相对位置 | [本周讲义](weeks/18_rope_and_extensions.md) | [06_modern_decoder.ipynb](../notebooks/core/06_modern_decoder.ipynb) | 07 |
| 12 | 19 | MQA 与 GQA | [本周讲义](weeks/19_mqa_gqa.md) | [06_modern_decoder.ipynb](../notebooks/core/06_modern_decoder.ipynb) | 08 |
| 13 | 20 | KV Cache 与逐 token 解码 | [本周讲义](weeks/20_kv_cache_decode.md) | [06_modern_decoder.ipynb](../notebooks/core/06_modern_decoder.ipynb) | 03 |
| 14 | 35 | 条件计算与 Top-k 路由 | [本周讲义](weeks/35_topk_routing.md) | [09_moe.ipynb](../notebooks/core/09_moe.ipynb) | 10 |
| 15 | 48 | 毕业项目与知识答辩 | [本周讲义](weeks/48_capstone_defense.md) | [11_inference_serving.ipynb](../notebooks/core/11_inference_serving.ipynb) | 可复现毕业报告、rubric 与口述答辩 |

使用方法：先运行 uv run llm-course lab，再按“学习单元 1 → 15”完成当前行的 Notebook，
然后运行 uv run llm-course exercises check <编号>；
没有编号的周次按 deliverable/rubric 验收。
维护者使用 uv run llm-course course check 核查 48 周资产闭环。
