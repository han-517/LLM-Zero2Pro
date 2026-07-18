# 核心代码模板与核查

starter 只保留函数签名、形状契约、分步提示和关键空缺；checker 公开边界、梯度和小规模数值 oracle，但不提供完整答案。清单由 `exercises/manifest.yaml` 驱动，CLI 与课程校验不再维护第二份编号表。

## 命令

```text
uv run llm-course exercises list
uv run llm-course exercises check 07
uv run llm-course exercises check rope
uv run llm-course exercises check all
```

第一次核查未填写 starter 会失败，这是学习起点，不是环境损坏。`all` 会检查所有模板，学习期间更适合只检查当前编号。

## 模板地图

| 编号 | 周次 | 别名 | 核心实现 |
|---|---:|---|---|
| 11 | 3 | autograd | 计算图、分支求和、重复 backward |
| 01 | 4 | softmax | 数值稳定 Softmax |
| 12 | 6–8 | neural-lm | Bigram、MLP、RNN 状态 |
| 06 | 10 | bpe | Byte BPE pair/merge |
| 02 | 12–13 | attention | 因果注意力与安全 mask |
| 13 | 14–15 | tiny-gpt | MHA、Block、TinyGPT |
| 09 | 16–17 | decoder | RMSNorm、SwiGLU |
| 07 | 18 | rope | 频率与二维旋转 |
| 08 | 19 | gqa | 分组 Query 与共享 KV |
| 03 | 20 | kv-cache | Prefill/decode 缓存账本 |
| 14 | 22–23 | data-pipeline | 去重、污染切分、packing |
| 15 | 24 | adamw-schedule | AdamW、warmup-cosine |
| 16 | 29–30, 32 | attention-frontiers | 分块、滑窗、线性注意力 |
| 17 | 31, 33 | mla-delta | latent KV 与 Delta 状态更新 |
| 10 | 35 | moe-router | Top-k 路由与 balance loss |
| 05 | 36 | moe-capacity | 容量、accepted/dropped 守恒 |
| 18 | 37–39 | moe-systems | 稳定性、共享专家、dispatch |
| 04 | 40 | sft | next-token shift 与 response mask |
| 19 | 41–44 | posttraining | LoRA、DPO、组内优势 |
| 20 | 45–47 | inference-systems | 量化、分页缓存、推测解码 |
| 21 | 选修 | multimodal-bridge | patch/projector/文本接口 |

## 两类检查

- `uv run llm-course course check`：维护者检查 48/48 周讲义、Notebook、starter/checker、来源与文档契约；不把空白 starter 当成仓库失败。
- `uv run llm-course exercises check <ID>`：学员检查自己的填写；失败退出码可用于编辑器与 CI。

公开测试通过仍不是最终掌握。你还应能解释每个维度、至少一个非法输入、时间/空间复杂度和实现与论文完整算法之间的边界。
