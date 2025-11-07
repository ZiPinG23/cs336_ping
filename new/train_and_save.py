from cs336_basics.train_bpe_ref import train_bpe
import pickle
import sys
from multiprocessing import freeze_support


def main() -> None:
    """CLI entrypoint for training BPE and saving results."""
    input_path = sys.argv[1]
    vocab_size = int(sys.argv[2])
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    with open("bpe_vocab_merges.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
    print("Saved bpe_vocab_merges.pkl")


if __name__ == "__main__":
    # Necessary on Windows and some spawn-based start methods to
    # avoid recursive child process imports when using multiprocessing.
    freeze_support()
    main()
