
import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np



def create_co_matrix(corpus, vocab_size, window_size):
    corpus_size = len(corpus) # corpus_size 是对应语料的大小，比如，一个句子的长度。有重复计数。
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32) # vocab_size是词库的大小。无重复计数。

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - 1
            right_idx = idx + 1

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id][left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id][right_word_id] += 1

    return co_matrix


def create_ppmi_matrix(co_matrix, verbose=False, eps=1e-8):
    ppmi_matrix = np.zeros_like(co_matrix, dtype=np.float32)
    N = np.sum(co_matrix) # bug: 总的出现次数， 这不是多算了吗？ 应该是语料库的单词数吧？
    SUM = np.sum(co_matrix, axis=0) # 某个字的出现次数

    (n_row, n_col) = co_matrix.shape
    total = n_row * n_col # 计算这么多次
    cnt = 0
    for i in range(n_row):
        for j in range(n_col):
            pmi = np.log2(co_matrix[i, j] * N / (SUM[i] * SUM[j]) + eps)
            ppmi_matrix[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100 + 1) == 0:
                    print("create_ppmi: %.1f%% done" % (100 * cnt / total))
    
    return ppmi_matrix
    pass



def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    if query not in word_to_id:
        print(f"Unkown query: {query}")
        return
    
    query_word_id = word_to_id[query]
    query_vec = word_matrix[query_word_id]

    vocab_size = len(word_to_id)
    similaritys = np.zeros(vocab_size)
    for word_id in range(vocab_size):
        similaritys[word_id] = cos_similarity(word_matrix[word_id], query_vec)

    count = 0
    for word_id in (-1 * similaritys).argsort(): # 对数组进行排序，然后返回索引
        if id_to_word[word_id] == query:
            continue

        print(f"{count}: {id_to_word[word_id]} = {similaritys[word_id]}")

        count += 1
        if count >= top:
            return

    pass



def main():
    import matplotlib.pyplot as plt
    from dataset.hello import hello
    from preprocess import build_corpus
    
    text = hello
    corpus, word_to_id, id_to_word = build_corpus(text)
    print(corpus, word_to_id, id_to_word)

    vocab_size = len(word_to_id)
    co_matrix = create_co_matrix(corpus, vocab_size, 1)
    print(co_matrix)

    ppmi_matrix = create_ppmi_matrix(co_matrix)
    np.set_printoptions(precision=3) # 打印精度
    print(ppmi_matrix)

    # 降维
    U, S, V = np.linalg.svd(ppmi_matrix)
    print(U[0])
    U2 = U[0:2]
    for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
    plt.scatter(U[:, 0], U[:, 1])
    plt.show()


    # cx = co_matrix[word_to_id['i']]
    # cy = co_matrix[word_to_id['you']]
    # sim_socre = cos_similarity(cx, cy)
    # print(sim_socre)

    # query = "you"
    # most_similar(query, word_to_id, id_to_word, co_matrix, 5)
    
    
    pass

if __name__ == '__main__':
    main()
