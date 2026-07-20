# 第 41 周：LoRA、QLoRA 与可合并低秩适配

## 课程定位

全参数微调为每个任务保存并优化整套权重，成本很高。LoRA 冻结底座，把权重更新限制为低秩乘积；QLoRA 再把冻结底座以 4-bit 形式存储，在较高精度中计算并训练 adapter。本周从矩阵秩和参数量推导 LoRA，核验 merge/unmerge 等价，并明确“教学 LoRA 层”与包含 NF4、double quantization、paged optimizer 和专用 kernel 的完整 QLoRA 系统不是同一件事。

## 学习目标

学习者应能写出 `W'=W+(α/r)BA`，追踪 A/B 形状并计算可训练参数；冻结底座且验证只有 adapter 获得梯度；解释 B 零初始化为何让初始模型与底座一致；在 eval 模式测试 merge 前后输出、重复 merge 幂等与 unmerge 恢复；区分存储 dtype、反量化计算 dtype、梯度 dtype 和优化器状态，并说出 QLoRA 不等于“把 LoRA 参数量化”。

## 前置知识

需要线性层 `y=xW^T+b`、矩阵秩、梯度与量化基本概念。仓库 `LoRALinear` 包装 `nn.Linear(in,out)`：`A:[r,in]`、`B:[out,r]`，因此 `BA:[out,in]` 与底座 weight 同形。所有比较应固定输入、dtype 与模式；含 dropout 的 adapter 在 train 模式输出是随机的，不能直接声称 merge 等价。

## 核心直觉

微调后的巨大权重差值未必需要充满整个 `out×in` 空间。LoRA 假设任务相关更新可由 r 个中间方向表达：输入先投影到 r 维，再投影回输出维。底座保留通用能力，adapter 只存小更新。merge 是代数重排：推理前把 `BA` 加入 W，便不必额外走两层分支；unmerge 再减回。同一 adapter 反复误加会污染底座，因此状态和幂等测试与公式同样重要。

## 张量与数据契约

输入 `x:[...,Din]`，底座 `W:[Dout,Din]`、bias `[Dout]`，LoRA `A:[r,Din]`、`B:[Dout,r]`，输出保持 `[...,Dout]`。r 为正且通常远小于两端维度，scale=`alpha/r`。底座参数 `requires_grad=False`；A/B 为浮点可训练参数。若使用 dropout，只作用于 adapter 输入且 merge 必须在 eval。adapter checkpoint 还需记录 base model 精确版本、target modules、rank、alpha、dropout、bias 策略与 tokenizer。

QLoRA 中冻结底座权重的存储量化、反量化后的矩阵乘 dtype、LoRA 参数和梯度 dtype 是不同层次。NF4 针对近似正态权重设计，double quantization 进一步压缩量化常数，paged optimizers 处理显存峰值；单纯把 float 权重 round 到 16 个级别并训练 A/B，只是伪量化教学，不足以复现 QLoRA。

## 公式推导与算法机制

普通线性层参数量为 `Dout×Din`。LoRA 可训练量是 `r(Din+Dout)`；例如 Din=Dout=4096、r=8 时，全更新约 1678 万参数，LoRA 为 65536，约 0.39%。前向

`y = xW^T + (α/r)(xA^T)B^T + b`。

由于 `(BA)^T=A^TB^T`，可合并成 `W_merge=W+(α/r)BA`。若 B 初始化为 0，则初始 `BA=0`，模型输出与底座逐值相同，而 A 保留随机方向；第一步 B 能收到梯度，随后 A 也开始学习。rank 越大表示能力更强但显存和参数增加，最佳 r 依赖目标模块与数据，不能从单个小实验外推。

## 手算与数值例

令 `W=I_2`、r=1、`A=[1,2]`、`B=[[3],[4]]`、alpha=1，则 `BA=[[3,6],[4,8]]`，合并权重 `[[4,6],[4,9]]`。输入行向量 `x=[1,-1]`，分支计算底座输出 `[1,-1]`，低秩支路先 `xA^T=-1`，再乘 B 得 `[-3,-4]`，合计 `[-2,-5]`；直接用合并权重的转置也得到 `[-2,-5]`。若误写 `AB`，形状或语义都错。

## 最小代码实现

```python
import torch
from llm_from_scratch.post_training import LoRALinear

torch.manual_seed(41)
layer = LoRALinear(torch.nn.Linear(5, 3), rank=2, alpha=4.0)
x = torch.randn(4, 5)
with torch.no_grad():
    layer.b.normal_(0, 0.1)  # 让 adapter 不再是零更新

layer.eval()
before = layer(x)
base_snapshot = layer.base.weight.detach().clone()
layer.merge_()
merged = layer(x)
assert torch.allclose(before, merged, atol=1e-6)
layer.merge_()               # 重复 merge 不得再次累加
assert torch.allclose(merged, layer(x), atol=1e-6)
layer.unmerge_()
assert torch.allclose(layer.base.weight, base_snapshot, atol=1e-6)
assert all(not p.requires_grad for p in layer.base.parameters())
print(sum(p.numel() for p in layer.parameters() if p.requires_grad))
```

## 反例、常见误区与调试

反例一冻结了 base weight 却漏掉 bias；打印所有 `named_parameters` 的梯度状态。反例二 B、A 都初始化为零，两者在起始点可能互相阻断梯度；参考实现通常 A 随机、B 为零。反例三 train 模式含 dropout 时比较 merge，会把随机差异误判为代数错误。反例四多次 merge 累加更新；维护明确 `_merged` 状态并测试幂等。反例五更换底座 revision 或 tokenizer 后硬加载 adapter，shape 相同也可能语义失配。反例六宣称 LoRA 按参数比例降低全部显存；激活、底座前向和临时张量仍占空间。

调试先在 2×2 矩阵上手算 `BA`，再关闭 dropout、用 float64 比较输出；检查底座 grad 为 None、A/B grad 有限；保存 merge 前 weight 快照，测试 merge→unmerge 恢复。QLoRA OOM 时还需分开量化权重、激活、梯度、优化器和临时反量化 buffer，而不是只统计 checkpoint 字节。

## 主流工作与实现边界

LoRA 常应用于 q/k/v/o 或 MLP 投影，现代 PEFT 还包括 rsLoRA、DoRA、LoftQ 与多 adapter 合并；不同方法并非默认可互换。QLoRA 的关键是冻结量化底座和较高精度计算，而不是训练 int4 权重。课程 `LoRALinear` 验证低秩更新与 merge 语义，不实现 NF4 编码、double quantization、paged optimizer、bitsandbytes kernel、分布式保存或多 adapter 路由。

## 实验与 Notebook 对照

运行 `learning/labs/10_posttraining.ipynb` 的 LoRA 单元，打开 `learning/readings/interactive/training-and-alignment.html` 理解训练内存构成。使用 `learning/labs/starter/19_posttraining.py` 的 sequence log-prob/组优势作为后续衔接；LoRA 核心可对照 `src/llm_from_scratch/post_training.py`。实验比较 r=1/2/4/8 对一个已知矩阵差值的拟合误差、可训练参数与 merge 误差，先预测秩上界再运行。

## 验收标准

合格：仅 A/B 可训练，初始输出等于底座，merge 前后误差 `<1e-6`。良好：unmerge 恢复、重复 merge 幂等，能计算参数节省并解释 B 零初始化。优秀：报告 target modules、base revision 和四类 dtype，做 rank 消融并准确列出教学实现缺失的 QLoRA 组件。把 fake int4+LoRA 称作完整 QLoRA 或底座仍有梯度者不通过。

## 一手来源

- LoRA 原论文：https://arxiv.org/abs/2106.09685
- LoRA 作者官方实现：https://github.com/microsoft/LoRA
- QLoRA 原论文：https://arxiv.org/abs/2305.14314
- QLoRA 官方实现仓库：https://github.com/artidoro/qlora
- Hugging Face PEFT 官方 LoRA 与合并文档：https://huggingface.co/docs/peft/main/conceptual_guides/lora
