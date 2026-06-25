import torch
from torch import nn


class FeedForward(nn.Module):
    """
    ## FFN module
    前馈神经网络，两层全连接网络，d_model -> d_ff -> d_model
    # d_ff 一般是 d_model 的四倍
    # 通过is_gated参数可以选择是否使用门控机制，即GLU（Gated Linear Units）
    # activation参数可以选择不同的激活函数，默认是ReLU
    # bias1, bias2, bias_gate参数可以选择是否在对应的线性层中使用偏置项
    """

    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):
        """
        * `d_model` is the number of features in a token embedding，d_model是词嵌入的特征数量
        * `d_ff` is the number of features in the hidden layer of the FFN，d_ff是FFN隐藏层的特征数量
        * `dropout` is dropout probability for the hidden layer,隐藏层的dropout概率
        * `is_gated` specifies whether the hidden layer is gated,是否使用门控机制
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1) # 线形层,输入层->隐藏层,d_model->d_ff,bias1表示是否使用偏置
        # Layer one parameterized by weight $W_1$ and bias $b_1$
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2) # 线形层,隐藏层->输出层,d_ff->d_model,bias2表示是否使用偏置
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)  # 随机失活层,防止过拟合
        # Activation function $f$
        self.activation = activation  # 激活函数
        # Whether there is a gate
        self.is_gated = is_gated  # 是否使用门控机制
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight $V$ and bias $c$
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)  # 线形层,输入层->门控层,d_model->d_ff,bias_gate表示是否使用偏置
            # Gated-FFN(x)=Wo*​(σ(Wg*​x)⊙(Wv*​x))+b ,其中σ是激活函数，⊙是逐元素乘法，bias_gate表示是否使用偏置
    
    # define the forward pass,定义前向传播过程
    def forward(self, x: torch.Tensor):
        # f(x W_1 + b_1), 对输入x进行线性变换后再经过激活函数,f表示激活函数,layer1表示线性层
        g = self.activation(self.layer1(x))
        # If gated, f(x W_1 + b_1) ⊙ (x V + b) = g ⊙ (x V + b)
        if self.is_gated:
            x = g * self.linear_v(x)
        # Otherwise
        else:
            x = g
        # Apply dropout,  应用随机失活
        x = self.dropout(x)
        
        # 有门控机制时，输出为(f(x W_1 + b_1) ⊙ (x V + b)) W_2 + b_2 
        # 没有门控机制时，输出为f(x W_1 + b_1) W_2 + b_2
        # depending on whether it is gated
        return self.layer2(x)


class LocalEnhancedFeedForward(nn.Module):
    """
    FFN with a lightweight local mixing step.

    It keeps the standard Transformer FFN structure
    d_model -> d_ff -> d_model, but inserts a depthwise 2D convolution
    on patch tokens after the expansion layer so nearby patches can interact.
    """

    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True,
                 kernel_size: int = 3):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

        padding = kernel_size // 2
        self.local_mixer = nn.Conv2d(
            d_ff,
            d_ff,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_ff,
            bias=True,
        )

    @staticmethod
    def _infer_grid(token_count: int):
        side = int(token_count ** 0.5)
        if side * side == token_count:
            return side, side
        return None

    def _apply_local_mixing(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape

        grid = self._infer_grid(seq_len - 1)
        has_cls_token = grid is not None
        if not has_cls_token:
            grid = self._infer_grid(seq_len)

        if grid is None:
            return x

        grid_h, grid_w = grid
        if has_cls_token:
            cls_token = x[:, :1, :]
            patch_tokens = x[:, 1:, :]
        else:
            cls_token = None
            patch_tokens = x

        patch_tokens = patch_tokens.view(batch_size, grid_h, grid_w, hidden_dim).permute(0, 3, 1, 2)
        patch_tokens = self.local_mixer(patch_tokens)
        patch_tokens = patch_tokens.permute(0, 2, 3, 1).contiguous().view(batch_size, grid_h * grid_w, hidden_dim)

        if cls_token is not None:
            return torch.cat([cls_token, patch_tokens], dim=1)
        return patch_tokens

    def forward(self, x: torch.Tensor, enable_local_mixing: bool = True):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g

        if enable_local_mixing:
            x = self._apply_local_mixing(x)
        x = self.dropout(x)
        return self.layer2(x)
