import os
import mmap
from typing import List, Tuple, Dict
from collections import Counter
import multiprocessing as mp
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from tqdm import tqdm

GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.IGNORECASE
)

def gpt2_pretokenize(text: str) -> List[bytes]:
    """把文本切成预token（bytes），用于 byte-level BPE 的后续统计。"""
    return [m.group(0).encode('utf-8') for m in GPT2_PAT.finditer(text)]

# ====== 3) 子进程 worker：读块 -> 预分词 -> 计数（推荐返回计数以省内存）======
def _worker_pretokenize(args) -> Counter:
    file_path, start, end, special_tokens = args
    with open(file_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        try:
            chunk_bytes = mm[start:end]
            text = chunk_bytes.decode("utf-8", errors="ignore")
        finally:
            mm.close()

    # 构造正则分隔符，确保特殊 token 不被正则特殊字符影响
    special_pat = "|".join(re.escape(tok) for tok in special_tokens)
    # 用 re.split 分割文本，保留分隔符
    parts = re.split(f"({special_pat})", text)
    toks = []
    for part in parts:
        if part in special_tokens:
            continue  # 跳过特殊 token
        elif part:
            toks.extend(gpt2_pretokenize(part))
    return Counter(toks)

# ====== 4) 并行入口：返回全语料的预token计数 ======
def parallel_pretokenize(file_path: str, num_procs: int,
                         split_tok: bytes = b"<|endoftext|>", special_tokens: List[str] = None) -> Counter:
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_procs, split_tok)
    # 组装 (path, start, end, special_tokens)
    tasks: List[Tuple[str, int, int, List[str]]] = [
        (file_path, s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])
    ]
    with mp.Pool(processes=min(num_procs, len(tasks))) as pool:
        parts: List[Counter] = pool.map(_worker_pretokenize, tasks)
    # 归并各块的计数
    total = Counter()
    for c in parts:
        total.update(c)
    return total


def merge(counter: Counter, vocabsize: int, vocab: dict[int, bytes]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """基于统计的 BPE 合并算法（增量更新版本，效率更高）。
    输入：token 计数器、目标词汇表大小、初始词汇表（int->bytes）
    输出：扩展后的词汇表（int->bytes）、合并列表（(bytes,bytes)）
    实现要点：一次性计算所有 pair 频率，然后在每次 merge 时只更新受影响的 word 的 pair 计数（增量更新），避免每轮全量重算。
    """
    # 1. 把 counter 的 key（bytes）转为 tuple[bytes] 的 word 形式
    word_counts: Counter = Counter()
    for tok, freq in counter.items():
        word = tuple(bytes([b]) for b in tok)
        word_counts[word] += freq

    merges: list[tuple[bytes, bytes]] = []
    cur_id = len(vocab)
    need_merge_cnt = max(0, vocabsize - cur_id)
    if need_merge_cnt == 0:
        return vocab, merges

    # 2. 初始计算所有 bigram 的频率（只做一次）
    pair_freqs: Counter = Counter()
    for word, cnt in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_freqs[pair] += cnt

    # 3. 迭代需要的 merge 次数，使用增量更新策略
    for _ in tqdm(range(need_merge_cnt)):
        if not pair_freqs:
            break
        # 选出频率最高且字典序最大的 pair
        max_freq = max(pair_freqs.values())
        candidates = [p for p, v in pair_freqs.items() if v == max_freq]
        best_pair = max(candidates)

        merges.append(best_pair)
        vocab[cur_id] = best_pair[0] + best_pair[1]
        cur_id += 1

        # 找出包含 best_pair 的 word（只处理这些 word）
        words_need_update: dict[tuple[bytes], int] = {}
        for word, cnt in word_counts.items():
            for i in range(len(word) - 1):
                if (word[i], word[i + 1]) == best_pair:
                    words_need_update[word] = cnt
                    break

        # 对每个受影响的 word 做增量更新：先从 pair_freqs 中减去原始 word 的 pair 计数，
        # 再生成合并后的 new_word，加入到 word_counts，并把 new_word 的 pair 计数加回 pair_freqs
        for word, cnt in words_need_update.items():
            # 减去旧的 pair 计数
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                new_val = pair_freqs.get(pair, 0) - cnt
                if new_val > 0:
                    pair_freqs[pair] = new_val
                else:
                    pair_freqs.pop(pair, None)

            # 从 word_counts 中移除旧 word
            del word_counts[word]

            # 构造合并后的 new_word
            new_word_list: list[bytes] = []
            i = 0
            L = len(word)
            while i < L:
                if i + 1 < L and (word[i], word[i + 1]) == best_pair:
                    new_word_list.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word_list.append(word[i])
                    i += 1
            new_word = tuple(new_word_list)

            # 把 new_word 放回 word_counts
            word_counts[new_word] = word_counts.get(new_word, 0) + cnt

            # 把 new_word 中的 pair 计数加回 pair_freqs
            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + cnt

    return vocab, merges
    
def initial_vacab(
        special_tokens: list[str]
) -> dict[int, bytes]:
    vocabulary = {}
    vocabulary.update({i: special_tokens[i].encode("utf-8") for i in range(0, len(special_tokens))})
    vocabulary.update({i + len(vocabulary): bytes([i]) for i in range(256)})  
    
    return vocabulary    
    
def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes],list[tuple[bytes,bytes]]]:
    initial_vocab = initial_vacab(special_tokens) # 第一步初始化词汇表

    counter = parallel_pretokenize(input_path, num_procs=1, split_tok=b"<|endoftext|>", special_tokens=special_tokens) # 第二步预分词
    
    merged_vocab, merges = merge(counter, vocab_size, initial_vocab)
    return merged_vocab, merges
