from collections.abc import Callable, Iterable
from typing import Optional
import torch 
from torch import nn
from einops import einsum, rearrange
import numpy.typing as npt
import math
import numpy as np

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features,in_features),device=device,dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, std=std, a=-3*std, b=3*std)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight,"... in_features, out_features in_features -> ... out_features")

class Embedding(nn.Module):
    def __init__(self, num_embeddings : int, embedding_dim : int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.weight = nn.Parameter(torch.empty((self.vocab_size, self.d_model), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight =nn.Parameter(torch.ones(self.d_model,device = device,dtype = dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS = (x.pow(2).mean(dim = -1,keepdim= True) + self.eps).sqrt()
        out = x / RMS * self.weight
        return out.to(dtype=in_dtype)
        
class FFN(nn.Module):
    def __init__(self,d_model:int,d_ff,device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(self.d_model,self.d_ff,device,dtype)
        self.w2 = Linear(self.d_ff,self.d_model,device,dtype)
        self.w3 = Linear(self.d_model,self.d_ff,device,dtype)
        
        
    def _silu(self,x : torch.Tensor):
        return x * torch.sigmoid(x)
    
    def _glu(self,x:torch.Tensor):
        return self._silu(self.w1(x)) * self.w3(x)
    
    def forward(self, x:torch.Tensor) ->torch.Tensor:
        return self.w2(self._glu(x))
    
class Rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # 预计算cos/sin缓存（仅在首次初始化时计算，避免重复计算）
        if not hasattr(self, "cos_cached") or not hasattr(self, "sin_cached"):
            # 1. 计算频率矩阵：shape (d_k//2,)
            freqs_d = 1 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
            # 2. 计算位置矩阵：shape (max_seq_len,)
            pos_i = torch.arange(max_seq_len, device=device).float()
            # 3. 频率-位置外积：shape (max_seq_len, d_k//2)
            freqs = einsum(freqs_d, pos_i, "d_half, max_seq_len -> max_seq_len d_half")
            # 预计算cos和sin值（后续直接索引使用）
            self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
            self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)
                
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_odd = x[..., 1::2] # 奇数维度：索引1,3,5...
        x_even = x[..., ::2] # 偶数维度：索引0,2,4...
        cos_pos = self.cos_cached[token_positions]  # shape (seq_len, d_k//2)
        sin_pos = self.sin_cached[token_positions]  # shape (seq_len, d_k//2)
        x_rotated_even = x_even * cos_pos - x_odd * sin_pos
        x_rotated_odd = x_even * sin_pos + x_odd * cos_pos
        x_rotated = torch.zeros_like(x)
        x_rotated[..., ::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd
        return x_rotated

def softmax(x: torch.Tensor,dim:int):
    max_i = torch.max(x,dim=dim,keepdim=True).values
    x_exp = torch.exp(x - max_i)
    sum_exp = torch.sum(x_exp,dim=dim,keepdim=True)
    return x_exp / sum_exp
 
def scaled_dot_product_attention(
     q : torch.Tensor, # Q：(batch_size, ..., seq_len_q, d_k)
     k : torch.Tensor, # K：(batch_size, ..., seq_len_k, d_k)
     v : torch.Tensor, # V：(batch_size, ..., seq_len_k, d_v)
     masks : torch.Tensor = None# 掩码：(seq_len_q, seq_len_k)，True表示可关注
    ) -> torch.Tensor:
    d_k = q.shape[-1]
    attention_score = einsum(q,k,"... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") / (d_k ** 0.5)
    if masks is not None:
        attention_score = attention_score.masked_fill(~masks, float('-inf'))
    attention_weights = softmax(attention_score,dim=-1)
    output = einsum(attention_weights,v,"... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v")
    return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, d_model : int, 
        num_heads :int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # 每个头的维度（d_model需能被num_heads整除）
        self.d_v = d_model // num_heads # 简化设计：V的维度与Q/K一致
        # 1. Q/K/V投影层：将d_model映射为 num_heads × d_k（或d_v）
        self.q_proj = Linear(d_model, num_heads * self.d_k)
        self.k_proj = Linear(d_model, num_heads * self.d_k)
        self.v_proj = Linear(d_model, num_heads * self.d_v)
        # 2. 输出投影层：将多个头的结果拼接后映射回d_model
        self.output_proj = Linear(num_heads * self.d_v, d_model)
        # 3. 若传入theta和max_seq_len，初始化RoPE模块
        if theta is not None and max_seq_len is not None:
            self.rope = Rope(theta, self.d_k, max_seq_len)
    
    def forward(self, x: torch.Tensor,
        mask: torch.Tensor | None = None,
        token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # 输入：(batch_size, seq_len, d_model)
        *batch_dims, seq_len, _ = x.shape # 提取批次维度（如batch_size）
        x_q = self.q_proj(x) # (batch_size, seq_len, num_heads×d_k)
        x_k = self.k_proj(x) # (batch_size, seq_len, num_heads×d_k)
        x_v = self.v_proj(x) # (batch_size, seq_len, num_heads×d_v)
        
        # 拆分多头：(batch_size, num_heads, seq_len, d_k)
        x_q = rearrange(x_q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
        num_heads=self.num_heads, d_k=self.d_k)
        x_k = rearrange(x_k, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
        num_heads=self.num_heads, d_k=self.d_k)
        x_v = rearrange(x_v, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v",
        num_heads=self.num_heads, d_v=self.d_v)
    
        if hasattr(self,"rope"):
            # 2. 应用RoPE（若已初始化）
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            x_q = self.rope(x_q, token_positions)
            x_k = self.rope(x_k, token_positions)
        
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)) #上三角掩码
            
        attn_output = scaled_dot_product_attention(x_q, x_k, x_v, mask)
        
        # 5. 拼接多头结果并投影
        attn_output = rearrange(attn_output, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)",
        num_heads=self.num_heads, d_v=self.d_v)
        output = self.output_proj(attn_output) # 投影回d_model维度
        return output 
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
theta: float | None = None,
max_seq_len: int | None = None,):
        super().__init__()
        self.d_model = d_model
        self.nums_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = FFN(d_model=d_model,d_ff=d_ff)
        if theta is not None and max_seq_len is not None:
            self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        else:
            self.attn = MultiHeadSelfAttention(d_model, num_heads)
        
    def forward(self,
x: torch.Tensor,
mask: torch.Tensor | None = None,
token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x),mask=mask, token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
    
class TransformerLM(nn.Module):
    def __init__(self,
                vocab_size:int,
                context_length:int,
                num_layers:int,
                d_model: int,
                num_heads: int,
                d_ff: int,
                theta: float | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size,d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model,num_heads,d_ff,theta,context_length)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model,vocab_size)
    
    def forward(self,x : torch.Tensor)->torch.Tensor:
        # 输入：(batch_size, seq_len) → 输出：(batch_size, seq_len, vocab_size)
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x) # (batch_size, seq_len, vocab_size)
        return logits
        
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算批次上的平均交叉熵损失
    输入：
    inputs: (batch_size, ..., vocab_size) → 未归一化的logits
    targets: (batch_size, ...) → 真实token的索引
    输出：
    标量张量 → 批次平均损失
    """
    
    batch_size = inputs.shape[0]
    
    o_max = torch.max(inputs,dim=-1,keepdim=True).values
    
    o = inputs - o_max
    target_logits = o[torch.arange(batch_size), targets]
    logsumexp = torch.log(torch.sum(torch.exp(o), dim=-1))
    # 单个样本损失：-target_logit + logsumexp
    loss = -target_logits + logsumexp
    # 返回批次平均值
    return loss.mean(dim=0)

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate:{lr}")
        defaults = {"lr":lr}
        super().__init__(params, defaults)
    
    def step(self,closure:Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                
                state = self.state[param]
                t = state.get("t",0)
                grad = param.grad.data
                param.data -= lr / math.sqrt(t+1) * grad
                
                state["t"] = t + 1
        
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params,lr = 1e-3,betas = (0.9,0.99),eps = 1e-8,weight_decay = 0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate:{lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value:{eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0:{betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1:{betas[1]}")
        defaults = {"lr":lr,
                    "betas":betas,
                    "eps":eps,
                    "weight_decay":weight_decay}
        
        super().__init__(params, defaults)
    
    def step(self,closure:Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1,beta2 = group["betas"]
            weigh_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                # 初始化状态（首次更新时）
                t = state.get("t", 1) # 迭代次数（初始为1）
                m = state.get("m", torch.zeros_like(grad)) # 一阶矩估计（动量）
                v = state.get("v", torch.zeros_like(grad)) # 二阶矩估计
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                # 计算偏差修正后的矩估计
                lr_t = lr * (math.sqrt(1-beta2**t)/(1-beta1**t))
                # 参数更新（含权重衰减）
                p.data -= lr_t*m/(v**0.5+eps)
                p.data -= lr*p.data*weigh_decay
                
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
                
        return loss

def lr_cosine_scheduler(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    ) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + (1+math.cos((it - warmup_iters)/(cosine_cycle_iters - warmup_iters)*math.pi))/2*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate
    
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float,eps = 1e-6
):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    l2_norm = 0.0
    for g in grads:
        l2_norm += torch.sum(g**2)
    l2_norm = torch.sqrt(l2_norm)
    clip_coef = min(1.0,max_l2_norm/(l2_norm+eps))
    for g in grads:
        g.mul_(clip_coef) 
    
def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length - 1
    if max_start < 0 :
        raise ValueError("数据集长度小于指定context_length")
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    x_batch = []
    y_batch = []
    for s in starts:
        seq = dataset[s:s+context_length+1]
        x_batch.append(seq[:-1])
        y_batch.append(seq[1:])
    x = torch.tensor(x_batch,dtype=torch.long,device=device)
    y = torch.tensor(y_batch,dtype=torch.long,device=device)
    return (x,y)
    
import os
from typing import BinaryIO, IO

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """保存训练状态到文件"""
    checkpoint = {
        'model_state': model.state_dict(),       # 模型参数
        'optimizer_state': optimizer.state_dict(),  # 优化器状态
        'iteration': iteration,                  # 当前迭代次数
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """从文件加载训练状态并恢复模型和优化器"""
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['iteration']  # 返回保存时的迭代次数
