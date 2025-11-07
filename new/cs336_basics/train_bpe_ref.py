from collections import Counter
from multiprocessing import Pool
import regex as re
from tqdm import tqdm
from cs336_basics.pretokenization_example import find_chunk_boundaries

## initialize vocabulary step
def initialize_vocabulary(
        special_tokens: list[str]
) -> dict[int, bytes]:
    vocabulary = {}
    vocabulary.update({i: special_tokens[i].encode("utf-8") for i in range(0, len(special_tokens))})
    vocabulary.update({i + len(vocabulary): bytes([i]) for i in range(256)})  
    
    return vocabulary

## pre_tokenization step
def pre_tokenization(
        input: str, 
        special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    escaped_tokens = [re.escape(tok) for tok in special_tokens]
    split_pattern = "|".join(escaped_tokens) # 按special_tokens分割input
    match_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""") # 分割后匹配除去special_tokens中的word

    split_texts = re.split(split_pattern, input) # 得到分割后的文本，格式为list
    pre_tokens = {}
    for split_text in split_texts:
        for word in match_pattern.finditer(split_text):
            word_str = word.group(0).encode("utf-8")
            bytes_word_tuple = tuple(bytes([word]) for word in word_str)
            pre_tokens[bytes_word_tuple] = pre_tokens.get(bytes_word_tuple, 0) + 1 
    
    return pre_tokens

## merge_tools 
def get_pair_freq(
        word_counts: Counter[tuple[bytes]]
) -> Counter[tuple[bytes]]:
    freq_pair: Counter[tuple[bytes]] = {}
    for word, cnt in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            freq_pair[pair] = freq_pair.get(pair, 0) + cnt
    return freq_pair

## merge_tools 
def find_pair(
        freq_pair: Counter[tuple[bytes]]
) -> tuple[bytes]:
    
    max_value = max(freq_pair.values())
    max_pair = max([k for k, v in freq_pair.items() if v == max_value]) #再来一次max就可以获取字典序最大的pair
    return max_pair

## merge_tools
def get_merged_word(
        word: tuple[bytes], 
        cmp_pair: tuple[bytes]
) -> tuple[bytes]:
    new_word = [] # 存储merge后的word
    length, cur = len(word), 0
    while cur < length:
        if cur + 1 < length: # 当还能组成的pair时
            if (word[cur], word[cur + 1]) == cmp_pair: # 找到了可以merge的对象
                new_word.append(word[cur] + word[cur + 1])
                cur += 2
            else:
                new_word.append(word[cur])
                cur += 1    
        else:
            new_word.append(word[cur])
            cur += 1
    return tuple(new_word)

def merge_pre_tokens(
        dicts: list[Counter[tuple[bytes]]]
) -> Counter[tuple[bytes]]:
    merged_counter = Counter()
    for counter in dicts:
        merged_counter.update(counter)
    return merged_counter

## 多进程进行pre_tokenization
def parallel_pre_tokenization(
        file_path: str, 
        special_tokens: list[str], 
        num_workers: int = None
) -> Counter[tuple[bytes]]:
    params = []
    with open(file_path, 'rb') as f:
        boundary = find_chunk_boundaries(f, num_workers, special_tokens[0].encode("utf-8")) 
        for left, right in zip(boundary[:-1], boundary[1:]):
            f.seek(left)
            chunk = f.read(right - left).decode("utf-8", errors="ignore")
            params.append((chunk, special_tokens))
    with Pool(processes=num_workers) as pool:
        result_dicts = pool.starmap(pre_tokenization, params)

    return merge_pre_tokens(result_dicts)


def train_bpe(
        input_path: str, 
        vocab_size: int, 
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    ## setp1 initinalize vocabulary
    vocabulary: dict[int, bytes] = initialize_vocabulary(special_tokens)

    ## setp2 pre tokenization
    # file_path = "assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    word_counts = parallel_pre_tokenization(
            input_path,
            special_tokens,
            1
    )

    cur_id: int = len(vocabulary)
    merges: list[tuple[bytes, bytes]] = []
    ## step3 BPE merge
    need_merge_cnt: int = vocab_size - cur_id

    pair_freqs  = get_pair_freq(word_counts)

    for i in tqdm(range(need_merge_cnt)): # 迭代merge频次最高的byte-pair

        if not pair_freqs:
            break
        best_pair = find_pair(pair_freqs)
        merges.append(best_pair)
        vocabulary[cur_id] = best_pair[0] + best_pair[1]
        cur_id += 1

        # 找出所有需要更新的word
        words_need_update = {}
        for word, cnt in word_counts.items():
            if best_pair[0] in word and best_pair[1] in word:
                for i in range(len(word) - 1):
                    if (word[i], word[i + 1]) == best_pair:
                        words_need_update[word] = cnt
                        break

        # 更新word_counts
        for word, cnt in words_need_update.items():
            # 增量更新pair频率表
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) - cnt

            del word_counts[word]
            new_word = get_merged_word(word, best_pair)
            word_counts[new_word] = word_counts.get(new_word, 0) + cnt

            for i in range(len(new_word) - 1):
                pair = (new_word[i], new_word[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + cnt
    
    return vocabulary, merges
