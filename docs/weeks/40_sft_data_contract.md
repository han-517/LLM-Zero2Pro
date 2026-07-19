# 第 40 周：指令数据契约与 Response-only SFT

## 课程定位

预训练让模型续写文本，监督微调（SFT）把“系统约束—用户请求—助手回答”序列化为可学习的对话。本周的核心不是多跑几个 epoch，而是锁定 chat template、因果右移、回答掩码、padding 与截断这一整套数据契约。一个错一位的 mask 不一定报错，却可能让模型学习复述用户、预测角色标记或把 padding 当答案；因此 SFT 首先是数据正确性工程，之后才是优化问题。

## 学习目标

学习者应能把多轮对话确定地序列化为 token；区分 attention mask、assistant mask 与 labels；证明 `logits[:,t]` 预测 `tokens[:,t+1]`，因而 labels 与回答 mask 必须同步右移；实现 response-only loss 并验证改动 prompt 位置 logits 不影响损失；记录 BOS/EOS、角色标记、packing、padding side、截断方向和 loss normalization，且清楚 response-only 是一种训练选择而非唯一真理。

## 前置知识

需要 next-token 交叉熵、`[B,T,V]` logits、文档级数据切分和 tokenizer 基础。约定一条样本已经由固定模板得到 `input_ids:[T]`；本周不训练大型模型，只在 CPU 上核验对齐。训练前必须保存模板文本或版本、tokenizer 文件与 special token id，不能只写模型昵称，因为同一消息列表在不同模板下会得到不同 token 和 loss 边界。

## 核心直觉

因果 LM 每个位置像在看左侧上下文后猜右边一个 token。我们把 prompt 也放入上下文，是为了让回答条件化于用户请求；但 response-only SFT 通常不要求模型在 prompt 位置复现 prompt。assistant mask 因而标记“哪些真实 token 的预测错误需要付代价”，它标在 token 本身，而对应预测来自前一位置的 logit。assistant 起始标记、正文、EOS 是否计分必须明说；把 mask 机械地从字符串索引转换而不检查 token 边界，是最常见的静默错误。

## 张量与数据契约

未移位输入 `input_ids:[B,T]` 为 long、值域 `[0,V)`；`assistant_mask:[B,T]` 与其同形，值只能是 0/1；`attention_mask:[B,T]` 标记真实 token 而非 padding；模型输出 `logits:[B,T,V]` 为有限浮点。因果训练使用 `shift_logits=logits[:,:-1]`、`shift_labels=input_ids[:,1:]`、`shift_loss_mask=assistant_mask[:,1:]`，三者前两维同为 `[B,T-1]`。每条样本至少有一个有效回答 token；packing 时样本边界必须阻止跨样本注意力或在正确位置重置语义；被截断后若回答全失，应丢弃而非产生除零。

数据记录还应包括：system/user/assistant/tool 的序列化规则，是否训练全部 assistant 回合，BOS/EOS 是否重复添加，assistant 前缀是否计 loss，最大长度与截断侧，padding id 与 side，response token 的 batch 归一化方式。按“每个有效 token 平均”与“先每序列平均再 batch 平均”对长短回答权重不同，不能隐式切换。

## 算法机制与公式推导

令序列 token 为 `x_0,…,x_{T-1}`，回答指示量为 `m_t∈{0,1}`。response-only 因果损失是

`L = - (1/Σ_{t=1}^{T-1}m_t) Σ_{t=1}^{T-1} m_t log pθ(x_t | x_{<t})`。

模型在数组位置 `t-1` 给出预测 `x_t` 的分布，因此实现中的 mask 必须取 `m[:,1:]`。若只右移 labels、不右移 mask，原本标记第一个回答 token 的 1 会落到错误预测位置。labels 置 `-100` 与独立 loss mask 是两种等价接口的候选，但必须避免一边置 ignore、一边又选到 ignore label。

SFT 仍是最大似然，不直接优化事实性、安全或用户偏好。InstructGPT 的完整流程在 SFT 之后还包括偏好奖励模型和 PPO；本周只完成第一段可审计监督目标。

## 手算与数值例

设 token 为 `[BOS,user,A,assistant,B,EOS]`，assistant mask 为 `[0,0,0,0,1,1]`。模型的五个有效预测位置依次预测 `[user,A,assistant,B,EOS]`，同步右移后的 mask 是 `[0,0,0,1,1]`，所以只有预测 B 与 EOS 的两项进入损失。若忘记移 mask 得 `[0,0,0,0,1]`，会漏掉 B，只训练 EOS。若规定 assistant 角色 token 也计分，则原始 mask 与规则都要调整，不能靠猜。

## 最小代码实现

```python
import torch
from llm_from_scratch.post_training import causal_sft_loss, response_only_collator

batch = response_only_collator(
    [[1, 4, 5, 8, 6, 2], [1, 4, 7, 8, 9, 2]],
    [[0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 1, 1]],
    pad_token_id=0,
)
g = torch.Generator().manual_seed(40)
logits = torch.randn(2, 6, 10, generator=g)
loss = causal_sft_loss(logits, batch["input_ids"], batch["assistant_mask"])

# 只改不会计分的 prompt 预测位置，loss 应保持不变。
changed = logits.clone()
changed[:, :3] += 100.0
loss_changed = causal_sft_loss(changed, batch["input_ids"], batch["assistant_mask"])
assert torch.allclose(loss, loss_changed)
print(float(loss), batch["labels"].tolist())
```

## 反例、常见误区与调试

第一个反例是 chat template 已自动加入 BOS，却又在 tokenizer 外层添加一次；检查解码后的完整 token 和 special id。第二个是通过字符串查找“assistant”定位 mask，文本内容恰好包含相同词；应由模板生成过程或角色边界生成 token mask。第三个是截断从右侧切掉全部回答，只剩 prompt；collator 应拒绝。第四个是 packing 后允许前一个样本答案看见下一个样本，产生跨样本泄漏。第五个是只比较总 loss，不做 prompt 扰动不变性与第一个 answer token 敏感性测试。

调试按顺序打印：原消息、模板字符串、token/id、角色边界、未移位 mask、右移 labels/mask、逐 token loss。把一个错误 token 的 logit 人工提高，确认只有对应有效位置使损失下降。训练 loss 异常低时先查 label 泄漏与 mask，而不是庆祝收敛；训练后还要在相同基础能力、安全和格式题集上做回归。

## 主流工作与实现边界

现代开源训练器支持 completion-only 或 assistant-only loss、多轮模板与 packing，但参数名相同不代表模板语义相同。全 token SFT、只训练 assistant、只训练最后一轮都有人采用，选择取决于数据与目标。长上下文 packing、tool call 和隐藏 reasoning token 还需要更细的段类型契约。本周代码是 response-only CPU reference，不包含分布式训练、数据流式加载、混合精度、checkpoint 或生产级模板验证。

## 实验与 Notebook 对照

运行 `notebooks/core/10_posttraining.ipynb` 的 SFT shift/mask 单元，打开 `docs/interactive/training-and-alignment.html` 切换到 SFT。补完 `exercises/starter/04_sft_shift.py`，执行 `uv run llm-course exercises check 04`。实验中至少构造单轮、多轮、右侧 padding、回答被截断和 assistant 文本含角色词五类样本；先手写预期 shift，再运行代码。

## 验收标准

合格：shift 后 labels/mask 形状正确，prompt 扰动不改变 loss，starter 通过。良好：交付完整模板与特殊 token 契约，覆盖 EOS、padding、截断、多轮和空回答。优秀：能比较两种 loss normalization，构造 packing 泄漏测试，并对 SFT 前后运行相同能力/安全回归。只给出一个下降的训练曲线、没有逐 token 对齐证据者不通过。

## 一手来源

- InstructGPT 完整 SFT—RM—PPO 流程：https://arxiv.org/abs/2203.02155
- Llama 2 的 SFT、RLHF 与安全训练报告：https://arxiv.org/abs/2307.09288
- Transformers 官方 chat template 契约：https://huggingface.co/docs/transformers/chat_templating
- TRL 官方 `SFTTrainer` 文档：https://huggingface.co/docs/trl/sft_trainer
- OpenAI 官方 InstructGPT 项目说明：https://openai.com/index/instruction-following/
