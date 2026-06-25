import torch
from torch import nn
from .mha import MultiHeadAttention
from .ffn import FeedForward, LocalEnhancedFeedForward

class TransformerLayer(nn.Module):
    """
    <a id="TransformerLayer"></a>

    ## Transformer Layer,定义transformer层

    This can act as an encoder layer or a decoder layer. We use pre-norm.
    一个编码层或一个解码层的组合，使用预归一化，
    编码器和解码器共用一个结构，区别在于解码器中有一个源注意力子层，而编码器没有。
    """

    def __init__(self, *,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 src_attn: MultiHeadAttention = None,
                 feed_forward: FeedForward,
                 local_ffn_first_layer_only: bool = False,
                 dropout_prob: float):
        """
        # 多头注意力机制和前馈网络组成的transformer层
        * 相当于一层transformer的编码器层和解码器层
        * `d_model` is the token embedding size，特征维度
        * `self_attn` is the self attention module，自注意力模块
        * `src_attn` is the source attention module (when this is used in a decoder),源注意力模块（当在解码器中使用时）
        * if this is `None`, then this layer acts as an encoder layer，如果等于None，则该层作为编码器层
        * `feed_forward` is the feed forward module,前馈模块
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """ # 定义transformer层的组成部分
        super().__init__()
        self.size = d_model                            # 特征维度大小
        self.self_attn = self_attn                     # 自注意力模块
        self.src_attn = src_attn                       # 源注意力模块
        self.feed_forward = feed_forward               # 前馈模块
        self.local_ffn_first_layer_only = local_ffn_first_layer_only
        self.dropout = nn.Dropout(dropout_prob)        # 随机失活层
        self.norm_self_attn = nn.LayerNorm([d_model])  # 自注意力归一化层，归一化层,特征维度
        if self.src_attn is not None:                  # 如果源注意力模块不为空，decoder层
            self.norm_src_attn = nn.LayerNorm([d_model]) # 源注意力归一化层，特征维度
        self.norm_ff = nn.LayerNorm([d_model])         # 前馈归一化层，特征维度
        # Whether to save input to the feed forward layer
        self.is_save_ff_input = False                # 是否保存前馈层的输入

    def forward(self, *,
                x: torch.Tensor,                 # 输入张量，形状为 (batch_size, seq_len, d_model)
                mask: torch.Tensor,              # 掩码张量，形状为 (batch_size, seq_len, seq_len)，batch_size表示批量大小，seq_len表示序列长度
                layer_idx: int = None,
                src: torch.Tensor = None,        # 源张量，形状为 (batch_size, src_seq_len, d_model)，用于解码器层
                src_mask: torch.Tensor = None):  # 源掩码张量，padding mask，填充掩码，形状为 (batch_size, seq_len, src_seq_len)
        """  PART 1: Self Attention"""
        # 自注意力机制，在decoder和encoder中都使用，但是在decoder中应用掩码mask,通过参数mask实现
        # 当mask有值时，为decoder，encoder时？
        # Normalize the vectors before doing self attention
        # 归一化输入向量，然后进行自注意力计算，“每一层内部、每个子层前”做 LayerNorm
        # pre-norm预归一化
        z = self.norm_self_attn(x)
        # Run through self attention, i.e. keys and values are from self
        # 返回自注意力的结果，查询、键、值都来自输入张量x，掩码为mask，返回自注意力得分[batch_size, seq_len, d_model]
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        # Add the self attention results, 残差连接，将自注意力结果与输入相加,应用nn.Dropout防止过拟合
        x = x + self.dropout(self_attn)
        # 完成自注意力子层的计算，接下来是源注意力子层（如果有的话），在decoder中使用，然后是FFN前馈子层
        
        """  PART 2: Source Attention（可选，仅在decoder中使用） """
        # If a source is provided, get results from attention to source.
        # This is when you have a decoder layer that pays attention to 
        # encoder outputs
        if src is not None:
            # Normalize vectors
            z = self.norm_src_attn(x)   # 应用源注意力归一化
            # Attention to source. i.e. keys and values are from source
            # 计算源注意力，keys和values来自源,query来自PART1的输出，经过归一化，掩码为src_mask
            attn_src = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            # Add the source attention results，残差连接，将源注意力结果与PART1的输出相加，应用nn.Dropout防止过拟合
            x = x + self.dropout(attn_src)
        
        """  PART 3: Feed Forward Network，FFN前馈子层"""
        # Normalize for feed-forward,归一化前面的结果
        z = self.norm_ff(x)
        # Save the input to the feed forward layer if specified
        # 保存前馈层的输入，保存的是归一化后的结果，保存在ff_input属性中
        if self.is_save_ff_input:
            self.ff_input = z.clone() # 分配内存，保存归一化后的结果，保留gradient属性
        # Pass through the feed-forward network
        # 前馈网络，经过前馈层变换
        if self.local_ffn_first_layer_only and isinstance(self.feed_forward, LocalEnhancedFeedForward):
            ff = self.feed_forward(z, enable_local_mixing=(layer_idx == 0))
        else:
            ff = self.feed_forward(z)
        # Add the feed-forward results back
        # 残差连接，将前馈层结果与PART1(或PART2)的输出相加，应用nn.Dropout防止过拟合
        x = x + self.dropout(ff)
        # x.shape = [batch_size, seq_len, d_model]
        return x
