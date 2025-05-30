"""
在Python中下载和评估HellaSwag数据集。
https://github.com/rowanz/hellaswag

HellaSwag json条目示例:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: 数据集ID
activity_label: 此示例的ActivityNet或WikiHow标签
context: 有两种格式。完整的上下文在ctx中。当上下文以不完整的名词短语结尾时，比如ActivityNet，这个不完整的名词短语在ctx_b中，而在此之前的上下文在ctx_a中。这对于需要最后一个句子完整的模型（如BERT）很有用。但是，这从来不是必需的。如果ctx_b非空，则ctx与ctx_a加上一个空格再加上ctx_b相同。
endings: 4个结尾的列表。正确的索引由label给出（0,1,2或3）
split: train, val或test。
split_type: 如果活动标签在训练期间可见则为indomain，否则为zeroshot
source_id: 这个示例来自哪个视频或WikiHow文章

gpt2 (124M)
- eleuther harness报告准确率28.92%，标准化准确率31.14%（多选题风格）
- 此脚本: 10042 准确率: 0.2859 标准化准确率: 0.2955（完成风格）

gpt2-xl (1558M)
- eleuther harness报告准确率40.04%，标准化准确率50.89%（多选题风格）
- 此脚本: 10042 准确率: 0.3842 标准化准确率: 0.4893（完成风格）

HellaSwag的验证集总共有10,042个示例。
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    """从给定URL下载文件的辅助函数"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split):
    """下载HellaSwag到DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"正在下载 {data_url} 到 {data_filename}...")
        download_file(data_url, data_filename)

def render_example(example):
    """
    给定示例作为字典，将其渲染为三个torch张量：
    - tokens（上下文+完成的标记，大小为4xN，因为总是有4个候选）
    - mask（在候选完成区域为1，我们在此处评估似然性）
    - label（正确完成的索引，我们希望它具有最高的似然性）
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # 在C语言版本中重现此评估所需的数据
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # 收集所有标记
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # 注意：前缀" "因为GPT-2分词器
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # 在整理过程中必须小心，因为每行的标记数可能不同
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    # val中总共有10,042个示例
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate(model_type, device):

    torch.set_float32_matmul_precision('high') # 使用tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # model = torch.compile(model) # 可选择性地torch编译模型

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # 获取logits
        logits = model(tokens).logits
        # 在所有位置评估自回归损失
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # 现在仅获取完成区域（mask == 1）的平均损失，在每一行中
        shift_mask = (mask[..., 1:]).contiguous() # 我们必须移位mask，所以从最后一个提示标记开始
        masked_shift_losses = shift_losses * shift_mask
        # 求和并除以mask中1的数量
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # 现在我们对4个完成中的每一个都有一个损失
        # 损失最低的应该是最可能的
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # 累积统计信息
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} 标准化准确率: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # 调试：美观打印几个示例，以及每种情况下的损失
        if num_total < 10:
            print("---")
            print(f"上下文:\n {example['ctx']}")
            print(f"结尾:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (损失: {avg_loss[i].item():.4f}) {end}")
            print(f"预测: {pred_norm}, 实际: {label}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="要使用的模型类型")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="要使用的设备")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)
