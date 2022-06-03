import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np

def build_corpus(text:str):
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            id = len(word_to_id)
            word_to_id[word] = id
            id_to_word[id] = word
    
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word


def create_context_target(corpus, window_size):
    target = corpus[window_size: -window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cur_contexts = []
        for delta in range(-window_size, window_size+1):
            if delta == 0:
                continue
            cur_contexts.append(corpus[idx + delta])
        contexts.append(cur_contexts)

    return np.array(contexts), np.array(target)


    pass


def convert_one_hot(corpus, vocab_size):
    
    n_samples = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((n_samples, vocab_size), dtype=np.int32)
        for idx, word_idx in enumerate(corpus):
            one_hot[idx, word_idx] = 1

    elif corpus.ndim == 2:
        n_context = corpus.shape[1]
        one_hot = np.zeros((n_samples, n_context, vocab_size), dtype=np.int32)
        for idx_sample, word_ids in enumerate(corpus):
            for idx, word_id in enumerate(word_ids):
                one_hot[idx_sample, idx, word_id] = 1
    
    return one_hot

def main():
    from dataset.hello import hello

    corpus, word_to_id, _ = build_corpus(hello)
    
    contests_list, target_list = create_context_target(corpus, 1)
    print("corpus: ", corpus)
    print("contests_list: ", contests_list)
    print("target_list: ", target_list)

    vocab_size = len(word_to_id)
    context_one_hot = convert_one_hot(contests_list, vocab_size)
    print("context one hot: ", context_one_hot)
    
    target_one_hot = convert_one_hot(target_list, vocab_size)
    print("target one hot: ", target_one_hot)

    pass

if __name__ == '__main__':
    main()