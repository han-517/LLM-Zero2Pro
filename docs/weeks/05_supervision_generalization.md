# 第 5 周：监督学习、泛化与数据切分

## 课程定位

模型在训练集上记得很牢，并不代表学会了可迁移规律。语言数据的危险在于相邻滑窗高度重叠：若先把 token 流切成窗口再随机分组，几乎相同的上下文会同时进入训练与验证，形成安静的数据泄漏。本周把监督学习写成明确的数据生成与评估契约，建立 train/validation/test 三者的职责，并用学习曲线区分欠拟合、过拟合与分布漂移。

## 学习目标

学习者应能定义样本、特征、标签和损失；解释经验风险与真实风险的差别；按文档而不是按重叠窗口切分语言数据；只用训练集拟合参数、用验证集选择配置、让测试集保持一次性审计；识别训练/验证曲线的典型模式；记录去重、随机种子和数据版本，避免测试污染。

## 前置

需要第 4 周的交叉熵与训练循环。约定原始语料是文档序列 `documents: Sequence[str]`；先以固定 seed 打乱文档并分成三个不重叠集合，再在每个集合内部 tokenize、构造 `[context,target]`。切分单位必须与潜在重复相关性匹配：同一网页的段落、同一代码仓库的文件或同一会话的轮次通常应归到同一侧。

## 自洽直觉

训练集像练习题，验证集像模拟考，测试集像封存的正式考卷。反复看正式考卷并据此调参，相当于把测试信息通过人的决策通道泄漏进模型，即使文件从未参与梯度。泛化间隙 `validation_loss-training_loss` 是诊断信号，不是越小越好：两者都高常是欠拟合，两者训练低而验证高常是过拟合，两者先降后验证升提示应早停或增强数据。若验证分布本身与部署不同，曲线再漂亮也不能代表真实效果。

## 张量/数据契约

每篇 token 文档是一维 long 张量 `[L_i]`，值域 `[0,V)`；固定窗口样本为 `contexts:[N,C]` 和 `targets:[N]`，其中目标是紧随窗口的 token。任何一个窗口不得跨文档边界。训练、验证、测试文档标识集合必须两两不交，重复内容应按哈希或近重复规则审计。所有基于全语料估计的变换——词表、标准化统计、过滤阈值——若会利用标签或测试频率，也必须只在训练侧拟合后应用到其他 split。

## 机制与推导

训练目标是经验风险 `R_train(θ)=1/N Σ_i ℓ(f_θ(x_i),y_i)`；真正关心的是未知数据分布上的期望风险 `R(θ)=E_{(x,y)~P}[ℓ]`。两者差值来自有限采样、模型选择和分布变化。验证集近似用于比较超参数 `h`：先对每个 h 在 train 求 `θ_h`，再选 `argmin_h R_val(θ_h)`。因为选择过程已经使用 validation，最终无偏审计需要未参与选择的 test。若在 test 上尝试 100 个配置后报告最好一个，测试集也变成了验证集。

重叠窗口泄漏可定量看出：长度 6 的 token 流、窗口 C=4 产生 `[t0..t3]→t4` 与 `[t1..t4]→t5`，输入有 3/4 重合。随机将两者分到 train/val，验证几乎是在识别训练片段。正确做法是先在文档层切分，使同源窗口不跨集合。

## 手算/数值例

有 10 篇文档，按 80/10/10 目标至少应得到 8/1/1 篇；但仅有 3 篇时仍必须保证 1/1/1，而不能让验证或测试为空。假设模型 A 的 train/val NLL 为 1.0/1.1，模型 B 为 0.2/1.8：B 训练更好却泛化更差。若测试 NLL 只在最终对 A 评一次得到 1.12，这是有效审计；若看到 1.12 后换 B、再改 tokenizer 并重复测试，测试数值已进入决策链，不能再称无偏。

## 最小可运行代码

```python
import torch
from llm_from_scratch.neural_lm import make_document_windows, split_documents

documents = [f"doc-{i}: abcdefgh" for i in range(10)]
split = split_documents(documents, seed=2026)
assert set(split.train).isdisjoint(split.validation)
assert set(split.train).isdisjoint(split.test)

# 示例把每个字符映为小整数；真实项目应固定 tokenizer 版本。
encoded = [torch.tensor([ord(ch) % 16 for ch in doc]) for doc in split.train]
contexts, targets = make_document_windows(encoded, context_size=4)
assert contexts.shape == (targets.numel(), 4)
print(len(split.train), len(split.validation), len(split.test), contexts.shape)
```

## 反例与调试

反例一是全语料拼接后切窗口再随机 split；相邻样本泄漏。反例二是先在全数据训练 tokenizer 或选择清洗阈值，再声称测试完全独立。反例三是每次实验更换 split seed，配置比较混入数据难度差异。反例四只看最后一个 epoch，忽略验证最低点与过拟合趋势。反例五按行随机切代码或对话，让同仓库、同会话内容跨集合。

调试应输出文档 ID 集合交集、内容哈希重复率、每侧 token 数和目标频率；人工抽样最近邻文本；将训练标签打乱，确认模型不能取得异常低验证损失。若训练和验证损失都不降，先检查标签 shift、学习率和模型容量；若只有验证恶化，检查泄漏、重复、过拟合和分布差异，而不是立即加大模型。

## 主流工作与边界

大规模预训练强调数据来源、去重、污染检测和数据卡；但“去重”没有唯一完美定义，精确哈希抓不到改写，模糊匹配又可能误删。Scaling laws 描述给定分布和预算下的经验趋势，不保证跨域部署。现代 benchmark 污染尤其难审计，公开测试文本可能进入网页语料。本周只建立离线监督 split；时间切分、在线 A/B、人类偏好和安全红队需要额外协议。

## 对应 Notebook、互动图与 starter

运行 `notebooks/core/02_neural_language_models.ipynb` 的数据切分与学习曲线部分，打开 `docs/interactive/foundations-lab.html` 的上下文模型视图。核心实现位于 `src/llm_from_scratch/neural_lm.py` 的 `split_documents`、`make_document_windows`；本周没有独立 starter，第 7–8 周共用 `exercises/starter/12_neural_lm.py`。

## 实验

实验一构造 12 篇带文档 ID 的短语料，比较“先窗口后随机切”和“先文档切后窗口”的最近邻重合率。实验二训练一个足以记忆训练集的小模型，记录每 10 step 的 train/val loss，并标注最佳 validation checkpoint。实验三故意在 test 上选学习率，写出信息泄漏路径，然后重置一个从未查看的新 holdout 才做最终审计。

## 验收 rubric

合格：三个文档集合不交、窗口不跨边界，能解释各 split 职责。良好：能诊断三类学习曲线，交付数据版本、seed、去重与选择协议。优秀：能设计针对近重复、tokenizer 拟合和测试调参的泄漏检查，并清楚说明验证集也会因反复选择而过拟合。随机窗口切分或用测试集调参者不通过。

## 一手来源

- The Pile 数据集论文与数据构成：https://arxiv.org/abs/2101.00027
- The Pile 官方代码与数据工具：https://github.com/EleutherAI/the-pile
- Deep Double Descent 原论文：https://arxiv.org/abs/1912.02292
- 神经语言模型 scaling laws 原论文：https://arxiv.org/abs/2001.08361
- PyTorch 官方可复现说明：https://docs.pytorch.org/docs/stable/notes/randomness.html
