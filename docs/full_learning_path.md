<!-- 由 uv run llm-course course path --weeks 48 --write 生成，请勿手工维护表格。 -->
# 48 周完整路线

这份路径由 course/roadmap.yaml 聚合 course.yaml 与 stages/。每周都按讲义、CPU Notebook、
starter/研究产出、自动核查或 rubric、一手来源形成闭环。
互动实验统一入口：[交互式实验](interactive/index.html)；代码练习清单：
[exercises/manifest.yaml](../exercises/manifest.yaml)。二者都由课程健康检查纳入契约。

| 周 | 主题 | 讲义 | Notebook | Starter / 产出 |
|---:|---|---|---|---|
| 1 | 命令行、Git、Python 与可复现环境 | [本周讲义](weeks/01_environment_reproducibility.md) | [00_START_HERE.ipynb](../notebooks/00_START_HERE.ipynb) | 环境诊断快照与复现命令 |
| 2 | 向量、矩阵与张量形状 | [本周讲义](weeks/02_tensor_shapes.md) | [01_shapes_and_autograd.ipynb](../notebooks/core/01_shapes_and_autograd.ipynb) | 三段张量代码的逐步形状表 |
| 3 | 导数、链式法则与自动微分 | [本周讲义](weeks/03_chain_rule_autograd.md) | [01_shapes_and_autograd.ipynb](../notebooks/core/01_shapes_and_autograd.ipynb) | 11 |
| 4 | 概率、Softmax、交叉熵与 PyTorch | [本周讲义](weeks/04_softmax_cross_entropy.md) | [01_shapes_and_autograd.ipynb](../notebooks/core/01_shapes_and_autograd.ipynb) | 01 |
| 5 | 监督学习、泛化与数据切分 | [本周讲义](weeks/05_supervision_generalization.md) | [02_neural_language_models.ipynb](../notebooks/core/02_neural_language_models.ipynb) | 无泄漏数据切分说明 |
| 6 | MLP、激活函数与优化 | [本周讲义](weeks/06_mlp_activations_optimization.md) | [02_neural_language_models.ipynb](../notebooks/core/02_neural_language_models.ipynb) | 12 |
| 7 | 词向量与神经语言模型 | [本周讲义](weeks/07_embeddings_neural_lm.md) | [02_neural_language_models.ipynb](../notebooks/core/02_neural_language_models.ipynb) | 12 |
| 8 | RNN、状态与序列瓶颈 | [本周讲义](weeks/08_rnn_state_and_sequence_bottleneck.md) | [02_neural_language_models.ipynb](../notebooks/core/02_neural_language_models.ipynb) | 12 |
| 9 | 文本、Unicode、字节与 token | [本周讲义](weeks/09_text_unicode_bytes_tokens.md) | [03_tokenization_and_bpe.ipynb](../notebooks/core/03_tokenization_and_bpe.ipynb) | 中英文字符、字节和 token 对照表 |
| 10 | 从零实现 BPE | [本周讲义](weeks/10_byte_bpe_from_scratch.md) | [03_tokenization_and_bpe.ipynb](../notebooks/core/03_tokenization_and_bpe.ipynb) | 06 |
| 11 | Embedding 与位置信息 | [本周讲义](weeks/11_embeddings_and_position.md) | [04_attention_mechanics.ipynb](../notebooks/core/04_attention_mechanics.ipynb) | 无位置编码时的顺序反例 |
| 12 | 缩放点积注意力 | [本周讲义](weeks/12_scaled_dot_product_attention.md) | [04_attention_mechanics.ipynb](../notebooks/core/04_attention_mechanics.ipynb) | 02 |
| 13 | 因果掩码与多头注意力 | [本周讲义](weeks/13_causal_mask_and_multihead_attention.md) | [04_attention_mechanics.ipynb](../notebooks/core/04_attention_mechanics.ipynb) | 02 |
| 14 | Transformer Block 与残差 | [本周讲义](weeks/14_transformer_block_and_residual.md) | [05_tiny_gpt.ipynb](../notebooks/core/05_tiny_gpt.ipynb) | 13 |
| 15 | 训练并采样 Tiny GPT | [本周讲义](weeks/15_train_and_sample_tiny_gpt.md) | [05_tiny_gpt.ipynb](../notebooks/core/05_tiny_gpt.ipynb) | 13 |
| 16 | Pre-Norm 与 RMSNorm | [本周讲义](weeks/16_rmsnorm_prenorm.md) | [06_modern_decoder.ipynb](../notebooks/core/06_modern_decoder.ipynb) | 09 |
| 17 | SwiGLU 与门控 MLP | [本周讲义](weeks/17_swiglu.md) | [06_modern_decoder.ipynb](../notebooks/core/06_modern_decoder.ipynb) | 09 |
| 18 | RoPE：用旋转编码相对位置 | [本周讲义](weeks/18_rope_and_extensions.md) | [06_modern_decoder.ipynb](../notebooks/core/06_modern_decoder.ipynb) | 07 |
| 19 | MQA 与 GQA | [本周讲义](weeks/19_mqa_gqa.md) | [06_modern_decoder.ipynb](../notebooks/core/06_modern_decoder.ipynb) | 08 |
| 20 | KV Cache 与逐 token 解码 | [本周讲义](weeks/20_kv_cache_decode.md) | [06_modern_decoder.ipynb](../notebooks/core/06_modern_decoder.ipynb) | 03 |
| 21 | 初始化、AdamW 与训练稳定性 | [本周讲义](weeks/21_initialization_stability.md) | [06_modern_decoder.ipynb](../notebooks/core/06_modern_decoder.ipynb) | 经典到现代组件逐项消融表 |
| 22 | 预训练数据从哪里来 | [本周讲义](weeks/22_data_governance.md) | [07_pretraining_systems.ipynb](../notebooks/core/07_pretraining_systems.ipynb) | 14 |
| 23 | 过滤、去重与数据混合 | [本周讲义](weeks/23_dedup_mixing_packing.md) | [07_pretraining_systems.ipynb](../notebooks/core/07_pretraining_systems.ipynb) | 14 |
| 24 | 优化器、Warmup 与学习率计划 | [本周讲义](weeks/24_adamw_schedules.md) | [07_pretraining_systems.ipynb](../notebooks/core/07_pretraining_systems.ipynb) | 15 |
| 25 | FLOPs、显存与混合精度 | [本周讲义](weeks/25_mixed_precision_resources.md) | [07_pretraining_systems.ipynb](../notebooks/core/07_pretraining_systems.ipynb) | 参数、激活和优化器内存预算 |
| 26 | 并行训练、检查点与故障恢复 | [本周讲义](weeks/26_parallel_checkpointing.md) | [07_pretraining_systems.ipynb](../notebooks/core/07_pretraining_systems.ipynb) | 并行拓扑与可恢复检查点方案 |
| 27 | Scaling Laws 与计算最优 | [本周讲义](weeks/27_scaling_and_evaluation.md) | [07_pretraining_systems.ipynb](../notebooks/core/07_pretraining_systems.ipynb) | Scaling Law 拟合与残差 |
| 28 | 可靠评测、去污染与故障诊断 | [本周讲义](weeks/28_reliable_evaluation.md) | [07_pretraining_systems.ipynb](../notebooks/core/07_pretraining_systems.ipynb) | 含去污染声明、区间与故障核查的评测卡 |
| 29 | FlashAttention 与 IO-aware 思维 | [本周讲义](weeks/29_flashattention.md) | [08_attention_frontiers.ipynb](../notebooks/core/08_attention_frontiers.ipynb) | 16 |
| 30 | 滑窗、块稀疏与全局 token | [本周讲义](weeks/30_sparse_attention.md) | [08_attention_frontiers.ipynb](../notebooks/core/08_attention_frontiers.ipynb) | 16 |
| 31 | MLA 与低维 KV 表示 | [本周讲义](weeks/31_mla.md) | [08_attention_frontiers.ipynb](../notebooks/core/08_attention_frontiers.ipynb) | 17 |
| 32 | 线性注意力的结合律与归一化状态 | [本周讲义](weeks/32_linear_attention.md) | [08_attention_frontiers.ipynb](../notebooks/core/08_attention_frontiers.ipynb) | 16 |
| 33 | Mamba-2、DeltaNet 与选择性状态更新 | [本周讲义](weeks/33_gated_deltanet.md) | [08_attention_frontiers.ipynb](../notebooks/core/08_attention_frontiers.ipynb) | 17 |
| 34 | 混合架构、DSA 与长上下文评测 | [本周讲义](weeks/34_hybrid_dsa_evaluation.md) | [08_attention_frontiers.ipynb](../notebooks/core/08_attention_frontiers.ipynb) | 混合架构长上下文公平消融 |
| 35 | 条件计算与 Top-k 路由 | [本周讲义](weeks/35_topk_routing.md) | [09_moe.ipynb](../notebooks/core/09_moe.ipynb) | 10 |
| 36 | 容量、token dropping 与负载均衡 | [本周讲义](weeks/36_capacity_and_balance.md) | [09_moe.ipynb](../notebooks/core/09_moe.ipynb) | 05 |
| 37 | Router z-loss、数值精度与稳定性 | [本周讲义](weeks/37_router_stability.md) | [09_moe.ipynb](../notebooks/core/09_moe.ipynb) | 18 |
| 38 | 共享专家、细粒度专家与 upcycling | [本周讲义](weeks/38_shared_experts_upcycling.md) | [09_moe.ipynb](../notebooks/core/09_moe.ipynb) | 18 |
| 39 | 专家并行、DeepSeekMoE 与现代变体 | [本周讲义](weeks/39_expert_parallel.md) | [09_moe.ipynb](../notebooks/core/09_moe.ipynb) | 18 |
| 40 | 指令数据与 SFT | [本周讲义](weeks/40_sft_data_contract.md) | [10_posttraining.ipynb](../notebooks/core/10_posttraining.ipynb) | 04 |
| 41 | LoRA 与 QLoRA | [本周讲义](weeks/41_lora_qlora.md) | [10_posttraining.ipynb](../notebooks/core/10_posttraining.ipynb) | 19 |
| 42 | 偏好、奖励模型与 PPO | [本周讲义](weeks/42_reward_model_ppo.md) | [10_posttraining.ipynb](../notebooks/core/10_posttraining.ipynb) | 19 |
| 43 | DPO：绕开奖励模型的偏好优化 | [本周讲义](weeks/43_dpo_preference_optimization.md) | [10_posttraining.ipynb](../notebooks/core/10_posttraining.ipynb) | 19 |
| 44 | GRPO、RLVR 与推理训练 | [本周讲义](weeks/44_grpo_rlvr_reasoning.md) | [10_posttraining.ipynb](../notebooks/core/10_posttraining.ipynb) | 19 |
| 45 | 权重量化与误差 | [本周讲义](weeks/45_quantization.md) | [11_inference_serving.ipynb](../notebooks/core/11_inference_serving.ipynb) | 20 |
| 46 | PagedAttention 与推测解码 | [本周讲义](weeks/46_paged_attention_continuous_batching.md) | [11_inference_serving.ipynb](../notebooks/core/11_inference_serving.ipynb) | 20 |
| 47 | 基准设计与公平比较 | [本周讲义](weeks/47_speculative_decoding_benchmarks.md) | [11_inference_serving.ipynb](../notebooks/core/11_inference_serving.ipynb) | 20 |
| 48 | 毕业项目与知识答辩 | [本周讲义](weeks/48_capstone_defense.md) | [11_inference_serving.ipynb](../notebooks/core/11_inference_serving.ipynb) | 可复现毕业报告、rubric 与口述答辩 |

使用方法：先运行 uv run llm-course lab，再按“原课程周 1 → 48”完成当前行的 Notebook，
然后运行 uv run llm-course exercises check <编号>；
没有编号的周次按 deliverable/rubric 验收。
维护者使用 uv run llm-course course check 核查 48 周资产闭环。
