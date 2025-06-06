# GPT-2训练脚本 - 从零开始训练一个GPT语言模型
# 支持单GPU和多GPU分布式训练
# 包含验证评估、HellaSwag测试和文本生成功能

# 启用CuDNN的Scaled Dot-Product Attention优化
import os
os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'

import math # 用于数学运算
import time # 用于时间相关操作
import inspect # 用于获取函数信息
from dataclasses import dataclass # 用于定义数据类
import torch # 用于深度学习
import torch.nn as nn # 用于定义神经网络层
from torch.nn import functional as F # 用于定义神经网络层
from hellaswag import render_example, iterate_examples  # 用于HellaSwag评估
# -----------------------------------------------------------------------------

# 多头自注意力机制 - GPT的核心组件
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 所有头的 key, query, value projections，批量处理
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # 正则化
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # 计算批处理中所有头的 query, key, values，并将头维度移到前面作为 batch 维度
        # nh 是"头数"，hs 是"头大小"，C（通道数）= nh * hs
        # 例如在 GPT-2 (124M) 中，n_head=12, hs=64，所以 nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 将所有头的输出并排重新组装
        # 输出投影
        y = self.c_proj(y)
        return y

# 前馈神经网络 - Transformer块的第二个主要组件  
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# Transformer块 - 包含注意力和MLP的完整单元
class Block(nn.Module):

    def __init__(self, config): 
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# GPT模型配置参数
@dataclass
class GPTConfig:
    block_size: int = 1024 # 最大序列长度
    vocab_size: int = 50257 # token 数量：50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # 层数
    n_head: int = 12 # 头数
    n_embd: int = 768 # embedding 维度

# 完整的GPT模型 - 包含embedding、transformer块和输出层
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享方案
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化参数
        self.apply(self._init_weights)

    # 初始化模型权重 - 使用标准的正态分布初始化
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # 模型前向传播 - 输入token序列，输出logits和损失
    def forward(self, idx, targets=None):
        # idx 的形状是 (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # 前向传播 token 和位置 embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # 位置 embeddings，形状 (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings，形状 (B, T, n_embd)
        x = tok_emb + pos_emb
        # 前向传播 transformer 的各个 blocks
        for block in self.transformer.h:
            x = block(x)
        # 前向传播最终的 layernorm 和分类器
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # 从HuggingFace加载预训练权重
    @classmethod
    def from_pretrained(cls, model_type):
        """从 huggingface 加载预训练的 GPT-2 模型权重"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'} # 确保模型类型是有效的
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head 和 n_embd 由 model_type 确定
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # GPT 模型 checkpoints 总是 50257
        config_args['block_size'] = 1024 # GPT 模型 checkpoints 总是 1024
        # 创建一个从头初始化的 minGPT 模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # 丢弃这个 mask/buffer，不是参数

        # 初始化一个 huggingface/transformers 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 复制时确保所有参数都对齐，名称和形状都匹配
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # 忽略这些，只是 buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # 同样，只是 mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # 基本上 openai checkpoints 使用 "Conv1D" 模块，但我们只想使用普通的 Linear
        # 这意味着我们在导入时必须转置这些权重
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 对需要转置的 Conv1D 权重进行特殊处理
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 普通复制其他参数
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    # 配置优化器 - 设置权重衰减和学习率
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # 从所有候选参数开始（需要梯度的）
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 创建优化器组。任何 2D 参数都会进行权重衰减，否则不会。
        # 即 matmuls + embeddings 中的所有权重张量衰减，所有 biases 和 layernorms 不衰减。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # 创建 AdamW optimizer 并在可用时使用 fused 版本
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken # 用于编码和解码文本
import numpy as np # 用于加载和处理数据

# 从numpy文件加载token数据
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # 视频后添加
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# 轻量级数据加载器 - 支持分布式训练的批次生成
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # 获取 shard 文件名
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    # 重置数据加载器到初始状态
    def reset(self):
        # 状态，从 shard 零开始初始化
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    # 获取下一个训练批次 - 返回输入和目标序列
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # 输入
        y = (buf[1:]).view(B, T) # 目标
        # 在张量中推进位置
        self.current_position += B * T * self.num_processes
        # 如果加载下一个 batch 会超出边界，推进到下一个 shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# HellaSwag 评估的辅助函数
# 接受 tokens、mask 和 logits，返回损失最低的补全的索引

# HellaSwag评估辅助函数 - 找出最可能的答案选项
def get_most_likely_row(tokens, mask, logits):
    # 在所有位置评估自回归损失
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # 现在只获取补全区域（mask == 1）的平均损失，每行
    shift_mask = (mask[..., 1:]).contiguous() # 我们必须移位 mask，所以从最后一个提示 token 开始
    masked_shift_losses = shift_losses * shift_mask
    # 求和并除以 mask 中 1 的数量
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # 现在我们有了 4 个补全的损失
    # 损失最低的应该是最可能的
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# 简单启动：
# python train_gpt2.py
# DDP 启动，例如 8 GPUs：
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# 运行训练循环
from torch.distributed import init_process_group, destroy_process_group  # 分布式训练初始化
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# 分布式训练设置 - 检测是否使用多GPU训练
ddp = int(os.environ.get('RANK', -1)) != -1 # 这是 ddp 运行吗？
if ddp:
    # 多GPU分布式训练配置
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # 这个进程将进行日志记录、checkpointing 等
else:
    # 单GPU或CPU训练配置
    ddp_rank = 0 # 分布式训练 rank
    ddp_local_rank = 0 # 本地 rank
    ddp_world_size = 1 # 世界大小
    master_process = True # 是否是主进程
    # 尝试自动检测设备
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# 视频后添加，pytorch 对其 device vs. device_type 区别很严格
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337) # 设置随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337) # 设置 CUDA 随机种子

enc = tiktoken.get_encoding("gpt2") # 使用 tiktoken 编码器

total_batch_size = 524288 # 2**19，~0.5M，token 数量
B = 64 # micro batch size
T = 1024 # 序列长度
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# 创建模型
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # 或从 OpenAI GPT-2 初始化
model.to(device)
use_compile = False # torch.compile 会干扰 HellaSwag 评估和生成。TODO 修复
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # 总是包含"原始"未包装模型

max_lr = 6e-4 # 最大学习率
min_lr = max_lr * 0.1 # 最小学习率
warmup_steps = 715 # 预热步数
max_steps = 19073 # 19,073 steps 约为 1 epoch，如果数据是 10B tokens，batch size 0.5M tokens

# 学习率调度函数 - 实现warmup和余弦衰减
def get_lr(it):
    # 1) warmup_iters steps 的线性 warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps # 线性预热
    # 2) 如果 it > lr_decay_iters，返回最小学习率
    if it > max_steps:
        return min_lr
    # 3) 在中间，使用余弦衰减到最小学习率
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff 从 1 开始到 0
    return min_lr + coeff * (max_lr - min_lr)

# 优化！
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# 创建我们将写入 checkpoints 和日志的日志目录
log_dir = "log" # 日志目录
os.makedirs(log_dir, exist_ok=True) # 创建日志目录
log_file = os.path.join(log_dir, f"log.txt") # 日志文件
with open(log_file, "w") as f: # 以写入模式打开以清空文件
    pass

for step in range(max_steps): # 训练循环
    t0 = time.time() # 开始时间
    last_step = (step == max_steps - 1) # 是否是最后一个步骤

    # 每250步进行验证损失评估
    if step % 250 == 0 or last_step:
        model.eval() # 设置模型为评估模式
        val_loader.reset() # 重置验证数据加载器
        with torch.no_grad(): # 禁用梯度计算
            val_loss_accum = 0.0 # 验证损失累加器
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # 可选择写入模型 checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # 如果您想更精确地恢复训练，您可能还想添加 optimizer.state_dict() 和
                # rng seeds 等
                torch.save(checkpoint, checkpoint_path)

    # 每250步进行HellaSwag常识推理评估
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # 仅处理 i % ddp_world_size == ddp_rank 的示例
            if i % ddp_world_size != ddp_rank:
                continue
            # 将示例渲染为 tokens 和 labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # 获取 logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # 在所有进程间缩减统计
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # 每250步生成文本样本来检查模型质量
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # 前向传播模型以获取 logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # 取最后位置的 logits
                logits = logits[:, -1, :] # (B, vocab_size)
                # 获取概率
                probs = F.softmax(logits, dim=-1)
                # 进行 top-k 采样，k=50（huggingface pipeline 默认）
                # 这里 topk_probs 变成 (5, 50)，topk_indices 是 (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # 从 top-k 概率中选择一个 token
                # 注意：multinomial 不要求输入总和为 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # 收集相应的索引
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # 附加到序列
                xgen = torch.cat((xgen, xcol), dim=1)
        # 打印生成的文本
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # 执行一步训练优化
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # 视频后添加，这个字段也被前向传播使用
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # 我们必须缩放损失以考虑梯度累积，
        # 因为梯度只是在每次连续的 backward() 上相加。
        # 梯度的相加对应于目标中的 SUM，但
        # 我们想要 MEAN 而不是 SUM。在这里缩放损失，使其正确
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # 确定并设置此迭代的学习率
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # 等待 GPU 完成工作
    t1 = time.time()
    dt = t1 - t0 # 以秒为单位的时间差
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
