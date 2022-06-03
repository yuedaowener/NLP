import sys
import os
import numpy as np
sys.path.append('.')
sys.path.append('..')

from dataset import ptb
from statistic_method import most_similar, create_co_matrix, create_ppmi_matrix


corpus, word_to_id, id_to_word, = ptb.load_data('train')
vocab_size = len(word_to_id)

window_size = 2
print("Create co_matrix ...")
co_matrix = create_co_matrix(corpus, vocab_size, window_size)

print("Create PPMI matrix ...")
ppmi_matrix = create_ppmi_matrix(co_matrix, True)

wordvec_size = 100 # 降维之后的维度
try:
    from sklearn.utils.extmath import randomized_svd
    print("Fast SVD ...")
    U, S, V = randomized_svd(ppmi_matrix, n_components=wordvec_size, n_iter=5, random_state=None)
except ImportError:
    print("Slow SVD ...")
    U, S, V = np.linalg.svd(ppmi_matrix)
word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
    




# print(len(corpus))