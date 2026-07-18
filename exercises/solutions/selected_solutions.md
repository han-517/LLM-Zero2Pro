# 精选答案与验收方法

## GQA Cache

单层 K/V 元素数忽略 batch 为：

```text
2 * sequence * kv_heads * head_dim
```

MHA：`2*1024*8*4=65536`；GQA：`2*1024*2*4=16384`，理论元素数减少 4 倍。Query 只在当前前向使用，不需要为历史 token 持久保存。

## MoE 容量

```text
ceil(1.25 * 100 * 2 / 8) = ceil(31.25) = 32
```

若 200 个 assignment 都去同一专家，只有 32 个能进入该专家，其余要显式丢弃、改派或由无 dropping 的动态稀疏实现处理。

## 正确的 KV Cache 验收

关闭 dropout，把模块设为 `eval()`。完整序列输出与逐 token 输出应在浮点容差内一致。只比较最后一个 token 不够，因为早期 cache 错误可能被后续偶然掩盖。

## “能过拟合但仍错误”的例子

- 因果 mask 反了，模型偷看 target。
- input 与 target 没有错开，模型只复制当前 token。
- 训练和验证使用同一片数据。

因此过拟合测试必须与因果性、target shift 和独立验证测试配套。

