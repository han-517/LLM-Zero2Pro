# 第 24 周：AdamW、Warmup 与学习率计划

## 课程定位

第 21 周从稳定性角度认识 AdamW，本周把它变成可单步验证、可恢复的优化器与 scheduler 契约。重点是偏差修正、decoupled weight decay、optimizer step 与 microstep、warmup/cosine 边界连续性，以及 checkpoint 恢复后的完全一致。

## 学习目标

- 从公式实现 AdamW 单步并与 PyTorch 参数更新对照。
- 解释 bias correction、epsilon 位置和 decoupled decay。
- 实现线性 warmup 接 cosine/linear decay，并画出每个 optimizer step 的 LR。
- 恢复 optimizer、scheduler、scaler 与 step 计数后保持下一步一致。

## 前置

需要掌握 SGD、动量、指数移动平均、梯度累积和参数组。先复习第 21 周更新式，并能说明为什么 LayerNorm/RMSNorm scale 与 bias 常被排除 decay 是配方选择而非 AdamW 定义本身。

## 直觉

一阶矩像平滑后的行进方向，二阶矩像每个参数坐标的近期尺度；Adam 用后者调节不同坐标步长。训练刚开始，移动平均仍偏向零，bias correction 补偿短历史。warmup 不是优化器内部状态，而是外部控制学习率从小到大。scheduler 必须以真正参数更新次数计步；若每个 microbatch 都前进一步，梯度累积会把计划压缩数倍。

## 张量/数据契约

每个可训练参数 `p` 对应同 shape 的 FP32 `m,v` 和整数 step。梯度必须在 accumulation 完成、AMP unscale 和非有限检查之后进入更新。参数组记录 `lr,betas,eps,weight_decay,decay_exempt`。scheduler 输入是 `optimizer_step`，总步数基于训练 token、global batch tokens 和计划 epochs 推导。checkpoint 必须保存 optimizer state、scheduler state、step、数据迭代器位置与 scaler。

## 推导与机制

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\]

`m_hat=m_t/(1-beta1^t)`、`v_hat=v_t/(1-beta2^t)`。AdamW 先按等价顺序应用 `p←(1-lr·wd)p` 与自适应步。线性 warmup 可写 `lr_t=lr_peak·t/W`；之后 cosine：

\[
lr_t=lr_{min}+\frac{lr_{peak}-lr_{min}}2
(lr_{peak}-lr_{min})\frac{\cos(\pi u)}2,
\]

更清楚地写为 `lr_min + 0.5*(lr_peak-lr_min)*(1+cos(pi*u))`，`u=(t-W)/(S-W)` 并截到 `[0,1]`。边界应连续。

## 数值例

第一步 `g=2,beta1=0.9,beta2=0.99`，则 `m=0.2,v=0.04`，偏差修正后 `m_hat=2,v_hat=4`，忽略 epsilon 的归一化方向为 1。若 `p=3,lr=0.1,wd=0.01`，decay 把参数乘 0.999，再减 0.1。若累积 4 个 microbatch，总计划 1000 次参数更新，scheduler 只能走 1000 步而不是 4000 步。

## 最小代码

```python
def warmup_cosine(step, warmup, total, peak, minimum):
    if not 0 <= step <= total or not 0 <= warmup < total:
        raise ValueError("step/warmup/total 契约非法")
    if step < warmup:
        return peak * (step + 1) / warmup if warmup else peak
    u = min(1.0, (step - warmup) / (total - warmup))
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * u))
```

课程单步 AdamW 是 oracle，真实训练使用 PyTorch fused/foreach 或分片 optimizer。不同实现对 epsilon、step 起点、capturable tensor 和 decay 顺序可能有细节差异，必须以所用官方 API 为准并保存版本。

## 反例与调试

把 L2 项加入梯度会同时污染 `m,v`，不是 AdamW。忘记 bias correction 时早期步长偏小。warmup 为 0 的除零、warmup 结束处 LR 跳变、最后一步没有到 `lr_min` 都应由边界测试捕获。恢复 checkpoint 后若 scheduler 先 step 一次，LR 会错一位。梯度累积中对每个 microbatch 清零或 step 会改变有效 batch 和训练轨迹。

## 主流工作与证据等级

Adam 与 AdamW 原论文提供算法基础；OLMo、LLaMA、OpenELM 等公开训练报告展示 `beta2≈0.95`、cosine、warmup 与 decay 的常见组合，属于公开配方证据。超参数与 batch、模型宽度、token 预算共同作用，不能把一个报告的 peak LR 直接搬到不同规模。框架文档是当前 API 真值，论文不是版本化调用说明。

## Notebook、互动图与 starter

在 `notebooks/core/07_pretraining_systems.ipynb` 画 LR 与参数范数曲线；完成 starter `15` 的 AdamW 单步和 schedule。`docs/interactive/training-and-alignment.html` 用于连接训练内存与目标，本周数值验收以单步 tensor oracle 为主。

## 实验

随机生成参数/梯度，比较自写与 `torch.optim.AdamW` 连续 20 步，覆盖多个 parameter group、zero grad 与 decay exempt。训练 Tiny GPT 比较无 warmup、连续 warmup+cosine、故意错一位 schedule；中途保存恢复，核对下一步参数逐元素相等。记录 LR、grad norm、参数 norm 和 loss。

## 验收 rubric

- 35%：AdamW 多步与框架 oracle 一致。
- 25%：scheduler 边界、累积 step 和恢复测试完整。
- 25%：训练实验固定 global batch tokens 与总更新数。
- 15%：明确教学 optimizer、fused/foreach、sharded 实现边界。

## 一手来源

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [PyTorch AdamW 官方文档](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
