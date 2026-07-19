<!-- 由 uv run llm-course course path --weeks 15 --write 生成，请勿手工维护表格。 -->
# 15 周核心路线

这份路径由 course/roadmap.yaml 聚合 course.yaml 与 stages/。每周都按讲义、CPU Notebook、
starter/研究产出、自动核查或 rubric、一手来源形成闭环。
互动实验统一入口：docs/interactive/index.html；代码练习清单：
exercises/manifest.yaml。二者都由课程健康检查纳入契约。


| 周 | 主题 | Notebook | Starter / 产出 |
|---:|---|---|---|
| 1 | 命令行、Git、Python 与可复现环境 | 00_START_HERE.ipynb | 环境诊断快照与复现命令 |
| 2 | 向量、矩阵与张量形状 | 01_shapes_and_autograd.ipynb | 三段张量代码的逐步形状表 |
| 3 | 导数、链式法则与自动微分 | 01_shapes_and_autograd.ipynb | 11 |
| 4 | 概率、Softmax、交叉熵与 PyTorch | 01_shapes_and_autograd.ipynb | 01 |
| 7 | 词向量与神经语言模型 | 02_neural_language_models.ipynb | 12 |
| 10 | 从零实现 BPE | 03_tokenization_and_bpe.ipynb | 06 |
| 12 | 缩放点积注意力 | 04_attention_mechanics.ipynb | 02 |
| 13 | 因果掩码与多头注意力 | 04_attention_mechanics.ipynb | 02 |
| 15 | 训练并采样 Tiny GPT | 05_tiny_gpt.ipynb | 13 |
| 16 | Pre-Norm 与 RMSNorm | 06_modern_decoder.ipynb | 09 |
| 18 | RoPE：用旋转编码相对位置 | 06_modern_decoder.ipynb | 07 |
| 19 | MQA 与 GQA | 06_modern_decoder.ipynb | 08 |
| 20 | KV Cache 与逐 token 解码 | 06_modern_decoder.ipynb | 03 |
| 35 | 条件计算与 Top-k 路由 | 09_moe.ipynb | 10 |
| 48 | 毕业项目与知识答辩 | 11_inference_serving.ipynb | 可复现毕业报告、rubric 与口述答辩 |

使用方法：先运行 uv run llm-course lab，完成当前周 Notebook，
再运行 uv run llm-course exercises check <编号>。维护者使用
uv run llm-course course check 核查 48 周资产闭环。
