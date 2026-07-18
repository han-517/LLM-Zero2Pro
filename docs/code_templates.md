# 核心代码模板与核查

这套练习把“阅读参考实现”和“自己完成核心代码”分开：`exercises/starter/` 保留函数签名、输入输出形状、分步提示和空缺实现；`exercises/checks/` 只包含公开行为测试，不包含答案。

## 三种命令

在仓库根目录运行：

```powershell
# 查看编号、周次、主题和当前填写状态
uv run llm-course exercises list

# 按编号或别名核查单题
uv run llm-course exercises check 07
uv run llm-course exercises check rope

# 核查全部模板
uv run llm-course exercises check all
```

模板初始包含 `raise NotImplementedError`，所以第一次运行核查应当失败。这代表练习尚未填写，不代表 Python、PyTorch 或 pytest 安装损坏。

## 模板地图

| 编号 | 周次 | 模板 | 核心空缺 | 核查重点 |
|---|---:|---|---|---|
| 01 | 4 | `01_stable_softmax.py` | 减最大值、指数与归一化 | 大 logits、平移不变性、均匀分布 |
| 02 | 12–13 | `02_causal_attention.py` | 缩放点积、因果 mask、聚合 | PyTorch oracle、梯度、未来隔离 |
| 03 | 20 | `03_kv_cache_budget.py` | Prefill/decode 账本、KV 元素数 | 缓存与重算工作量、K/V 双份存储 |
| 04 | 40 | `04_sft_shift.py` | labels 与 answer mask 同步右移 | 两个 batch、形状、输入不变 |
| 05 | 35–36 | `05_moe_capacity.py` | 容量、accepted/dropped | Top-k assignment、向上取整、守恒 |
| 06 | 9–10 | `06_byte_bpe.py` | pair 计数、选择、合并 | 重叠计数、确定性 tie-break、非重叠合并 |
| 07 | 21–22 | `07_rope.py` | 频率、二维旋转、维度交错 | 范数不变、position=0、奇数维拒绝 |
| 08 | 23–24 | `08_grouped_query_attention.py` | 分组 Query、共享 KV、因果聚合 | 显式 oracle、cache 对齐、梯度 |
| 09 | 21–22 | `09_modern_decoder.py` | RMSNorm、SwiGLU | 不减均值、门控顺序、反向传播 |
| 10 | 35–36 | `10_moe_router.py` | Top-k 路由、balance loss | 权重重归一、均衡基线、坍缩惩罚 |

## 推荐学习循环

1. 学对应讲义或操作交互图，先写下公式和每个张量形状。
2. 打开一个 starter，只填写 `TODO`，不要同时打开 `src/llm_from_scratch/`。
3. 先运行单题核查，从第一条失败信息构造最小反例。
4. 公开测试通过后，自己再添加一个边界输入。
5. 口述时间复杂度、空间复杂度和至少一个常见错误。
6. 最后才对照参考实现，记录两种实现的差异，而不是直接覆盖自己的版本。

## 两类检查的边界

- `uv run llm-course course check`：验证 48 周清单、论文目录、模板资产、参考实现和必修 Notebook 是否健康。它故意不执行未完成的 starter。
- `uv run llm-course exercises check ...`：加载你当前的 starter，执行独立公开测试；失败退出码可用于终端、编辑器任务或 CI。

公开测试通过不是最终掌握标准。特判固定输入、复制参考答案，或无法解释张量形状，都应视为尚未完成。需要分层提示时先看 [`exercises/hints.md`](../exercises/hints.md)，再看 `src/llm_from_scratch/`。
