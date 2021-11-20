import numpy as np

def build_corpus(text):
    # word_id_dict, 
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")

    word2id, id2word = {}, {}
    for word in words:
        if word not in word2id:
            new_id = len(word2id)
            word2id[word] = new_id
            id2word[new_id] = word

    corpus = np.array([word2id[wd] for wd in words])
    
    return corpus, word2id, id2word



def creat_contexts_target(corpus, window_size=1):
    target = corpus[window_size : -window_size]

    contexts = []
    for idx in range(window_size, len(corpus) - window_size):
        cur_context = []
        for delta in range(-window_size, window_size + 1):
            if delta == 0:
                continue # 恰好是target
            cur_context.append(corpus[idx + delta])
        contexts.append(cur_context)

    return np.array(contexts), np.array(target)



def to_one_hot(corpus, vocab_size):
    n_sample = corpus.shape[0] # 样本个数
    
    if corpus.ndim == 1:    # 处理样本的tag部分
        one_hot = np.zeros((n_sample, vocab_size), dtype=np.int32)
        for sample_idx, word_idx in enumerate(corpus):
            one_hot[sample_idx, word_idx] = 1
        
    elif corpus.ndim == 2:  # 处理样本的x部分
        n_context = corpus.shape[1] # 一个样本的长度是C，这个和窗口设置有关。window_size =1时, n_context= 2
        one_hot = np.zeros((n_sample, n_context, vocab_size), dtype=np.int32) # 一个样本的x部分， x的长度是C
        for sample_idx, word_idxs in enumerate(corpus):
            for context_idx, word_idx in enumerate(word_idxs):
                one_hot[sample_idx, context_idx, word_idx] = 1
    else:
        print("Wrong corpus.ndim")
        exit()
    
    return one_hot

            
        


    
    pass

    





def main():
    corpus = "You say goodbye and I say hi"
    ids, word2id, id2word = build_corpus(corpus)
    print(ids, word2id, id2word)


if __name__ == "__main__":
    main()