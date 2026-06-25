"""
---
title: Multi-Headed Attention (MHA),多头注意力机制的实现
summary: >
  This implements the Multi-Headed Attention used in transformers
  using PyTorch with explanations.
---

# Multi-Headed Attention (MHA)

This implements the Multi-Headed Attention used in transformers

Here is the [training code](basic/autoregressive_experiment.html) that uses a basic transformer
with MHA for NLP auto-regression.

[Here is an experiment implementation](basic/autoregressive_experiment.html) that trains a simple transformer.
"""

import math
from typing import Optional, List

import torch
from torch import nn


class PrepareForMultiHeadAttention(nn.Module):
    """
    <a id="PrepareMHA"></a>

    ## Prepare for multi-head attention，为多头注意力机制准备，将输入线性变换并拆分为多个头，从而实现多头注意力

    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        """
            *`d_model (int): 模型特征维度
            *`heads (int): 头数
            *`d_k (int): 每个头的特征维度，一般而言d_k=d_model/heads
            *`bias (bool): 是否使用偏置项
        """
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias) # 线性变换，d_model-> heads*d_k，bias表示是否使用偏置项
        # Number of heads，头数，外部参数
        self.heads = heads
        # Number of dimensions in vectors in each head，每个头的特征维度，外部参数
        self.d_k = d_k

        
    # define the forward pass，定义前向传播过程    
    def forward(self, x: torch.Tensor):
        # Input has shape,输入形状现在现在主流的代码中是`[batch_size, seq_len, d_model]` or `[batch_size, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the heads.用线性变换应用于最后一个维度，并将其拆分为多个头
        # 获取x的形状[:-1],即除了最后一个维度之外的所有维度，这里返回(batch_size, seq_len) or (batch_size),类型是元组
        head_shape = x.shape[:-1]   # 获取x的前面的维度的形状，head_shape = (batch_size, seq_len) or (batch_size)


        # Linear transform, 线性变换，将d_model变换为heads*d_k
        x = self.linear(x)

        # Split last dimension into heads，将最后一个维度拆分为多个头
        # x.view()的作用是将x重新塑形为指定的形状，这里将最后一个维度拆分为heads和d_k两个维度
        # *head_shape的*是Python 的解包（unpacking）语法，表示将head_shape元组中的元素展开作为单独的参数传递给view函数
        # 等价于x.view(batch_size, seq_len, self.heads, self.d_k)
        # *head_shape表示将head_shape元组中的元素展开作为单独的参数传递给view函数，这里head_shape是(batch_size, seq_len) or (batch_size)，所以展开后就是batch_size, seq_len或者batch_size
        x = x.view(*head_shape, self.heads, self.d_k)
        # 此时x的形状为 `[batch_size, seq_len, heads, d_k]` or `[batch_size, heads, d_k]`
        # 返回x
        return x


class MultiHeadAttention(nn.Module):
    r"""
    <a id="MHA"></a> 

    ## Multi-Head Attention Module 定义多头注意力机制模块

    This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.
    计算缩放点积多头注意力机制
    $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$

    In simple terms, it finds keys that matches the query, and gets the values of
     those keys.

    It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
    This is done to avoid large dot-product values causing softmax to
    give very small gradients when $d_k$ is large.

    Softmax is calculated along the axis of of the sequence (or time).
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        """
        * `heads` is the number of heads.heads表示头数
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        * `d_model` 是 `query`，`key` 和 `value` 向量中的特征数量
        * `dropout_prob` is dropout probability for attention weights. dropout_prob表示注意力权重的随机失活概率
        * `bias` specifies whether to use bias in linear transformations.表示是否在线性变换中使用偏置
        """
        super(MultiHeadAttention, self).__init__()

        # 定义类的属性
        # Number of heads
        self.heads = heads
        # Number of features per head
        self.d_k = d_model // heads  # d_k的计算，一般是d_model/heads,即每个头的特征维度,`//`表示整除


        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        # 变换`query`，`key`和`value`向量以实现多头注意力机制，d_model -> heads * d_k
        # PrepareForMultiHeadAttention模块的作用是将输入线性变换并拆分为多个头，从而实现多头注意力
        # query, key, value 形状shape=[batch_size, seq_len, heads, d_model]
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)

        # Softmax for attention along the time dimension of `key`
        # 计算`key`的时间维度上的注意力的Softmax
        self.softmax = nn.Softmax(dim=-1)

        # Output layer，输出层
        self.output = nn.Linear(d_model, d_model)
        # Dropout，随机失活层
        self.dropout = nn.Dropout(dropout_prob)
        # Scaling factor before the softmax, softmax之前的缩放因子
        self.scale = 1 / math.sqrt(self.d_k)

        # We store attentions so that it can be used for logging, or other computations if needed
        # 我们存储注意力机制，以便在需要时将其用于日志记录或其他计算。
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys
        ### 计算query和key之间的分数
        This method can be overridden for other variations like relative attention.
        """
        # 计算Q*K^T,Q成K的转置，用到爱因斯坦求和约定
        # Calculate Q K^T or S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}
        """
        # query shape: [batch_size, seq_len_q, heads, d_k] -> {b,i,h,d}
        # key shape: [batch_size, seq_len_k, heads, d_k] -> {b,j,h,d}   
        # scores shape: [batch_size, heads, seq_len_q, seq_len_k,] -> {b,h,i,j}
        相同字母的维度对齐并相乘，未出现在输出中的字母会被求和消掉,d没有出现在输出中，所以d维度会被求和消掉
        """
        return torch.einsum('bihd,bjhd->bhij', query, key)


    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        """
        Prepare attention mask for scaled dot-product attention.
        为缩放点积注意力准备注意力掩码,确定形状,负责reshape和broadcast
        Args:
            mask: Tensor, 形状
                [batch_size, seq_len_k] 或 [batch_size, seq_len_q, seq_len_k]
            query_shape: query tensor形状, e.g. [batch_size, seq_len_q, heads, d_k]
            key_shape: key tensor形状, e.g. [batch_size, seq_len_k, heads, d_k]

        Returns:
            mask: Tensor of shape [batch_size, 1, seq_len_q, seq_len_k]
                (ready to broadcast over heads)
        """
        batch_size = query_shape[0]
        seq_len_q = query_shape[1]
        seq_len_k = key_shape[1]

        # 情况 1：padding mask
        # mask: [batch_size, seq_len_k]
        if mask.dim() == 2:
            mask = mask[:, None, None, :]  # 第二个和第三个维度上添加新的维度,[b, 1, 1, k]
            mask = mask.expand(batch_size, 1, seq_len_q, seq_len_k)

        # 情况 2：显式 attention mask
        # mask: [batch_size, seq_len_q, seq_len_k]
        elif mask.dim() == 3:
            mask = mask[:, None, :, :]              # [b, 1, q, k]

        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

        # resulting mask has shape `[batch_size, 1, seq_len_q, seq_len_k]`，1广播到heads维度
        return mask

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        ### Forward pass for multi-head attention, 多头注意力机制的前向传播过程
        `query`, `key` and `value` are the tensors
        他们的形状 `[batch_size, seq_len, d_model]`.

        query: [B, Len_q, d_model]
        key:   [B, Len_k, d_model]
        value: [B, Len_k, d_model]
        mask:  可选，任意可被 prepare_mask 处理的形态,默认参数None
        """

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        batch_size, seq_len, _ = query.shape

        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[seq_len, batch_size, heads, d_k]`.
        # 线性变换，拆分为多个头，返回query,key,value的形状为`[batch_size, seq_len, heads, d_k]`
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # 计算注意力得分Q*K^T
        # 得到的scores的形状 `[batch_size, heads, seq_len_q, seq_len_k]`.
        scores = self.get_scores(query, key)

        # Scale scores 缩放,乘1/sqrt(d_k), {Q*K^T}/{sqrt{d_k}}
        scores *= self.scale


        # Apply mask
        if mask is not None:   # 如果mask不为空，应用掩码，掩码位置的值是-inf
            mask = self.prepare_mask(mask, query.shape, key.shape)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        # softmax 得到注意力权重,对key维度做softmax
        attn = self.softmax(scores)

        # Save attentions for any other calculations 
        self.attn = attn.detach()  # 保存注意力权重以供其他计算使用

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by values，乘以values
        # softmax{ {Q K^T}/{sqrt{d_k}} } * V
        x = torch.einsum('bhij,bjhd->bihd', attn, value)

        

        # Concatenate multiple heads
        # x = x.reshape(batch_size, seq_len, -1)
        # Concatenate multiple heads, 合并多头
        x = x.contiguous().view(batch_size, seq_len, self.heads * self.d_k)

        # Output layer,输出线性层，输出形状 `[batch_size, seq_len, d_model]`保持不变
        return self.output(x)


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟参数
    batch_size = 2
    seq_len = 5
    d_model = 64
    heads = 8
    
    
    print("successfully imported multi-headed attention module")
    # 创建模型
    mha = MultiHeadAttention(heads=heads, d_model=d_model)
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 这里的 mask 模拟 Padding Mask，最后两个 token 是 padding (0)
    mask = torch.ones(batch_size, seq_len)
    mask[:, -2:] = 0 
    
    # 前向传播 (Self-Attention: Q=K=V=x)
    output = mha(query=x, key=x, value=x, mask=mask)
    
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("✅ 多头注意力模块运行成功！")