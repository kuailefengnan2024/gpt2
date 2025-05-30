"""
FineWeb-Edu 数据集 (用于自监督预训练)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
下载并分词数据，将数据分片保存到磁盘。
简单运行：
$ python fineweb.py
将会将分片保存到本地目录 "edu_fineweb10B"。
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip 安装 datasets
from tqdm import tqdm # pip 安装 tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 每个分片1亿个tokens，总共100个分片

def tokenize(doc):
    # 对单个文档进行分词并返回uint16类型的numpy数组
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>'] # 文本结束标记
    tokens = [eot] # 特殊的<|endoftext|>标记分隔所有文档
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

if __name__ == '__main__':
    # 如果本地缓存目录不存在则创建
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # 下载数据集
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

    # 对所有文档进行分词并写入输出分片，每个分片包含shard_size个tokens（最后一个分片包含剩余的）
    nprocs = max(1, os.cpu_count()//2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # 预分配缓冲区来保存当前分片
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):

            # 当前分片是否有足够空间容纳新的tokens？
            if token_count + len(tokens) < shard_size:
                # 简单地将tokens追加到当前分片
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # 更新进度条
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # 写入当前分片并开始新的分片
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                # 将文档分割成适合此分片的部分；剩余部分进入下一个分片
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # 将当前文档的剩余部分填充到下一个分片
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # 将剩余的tokens写入最后一个分片
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
