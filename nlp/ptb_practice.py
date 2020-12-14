from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data("train")

print("말뭉치 크기 : ", len(corpus))
print("~30 : ", corpus[:30])
print("id_to_word : ", id_to_word[0])
print("id_to_word : ", id_to_word[1])
print("id_to_word : ", id_to_word[2])

print("word_to_id : ", word_to_id["cat"])
print("word_to_id : ", word_to_id["happy"])
print("word_to_id : ", word_to_id["lexus"])

import numpy as np
from common.util import most_similar, create_co_matrix, ppmi

window_size = 2
wordvec_size = 100
vocab_size = len(word_to_id)
print("========동시 발생수 계산========")
C = create_co_matrix(corpus, vocab_size, window_size)
print("========PPMI 계산========")
W = ppmi(C, verbose=True)
print("========SVD 계산========")

from sklearn.utils.extmath import randomized_svd

try:
    U, S, V = randomized_svd(W, n_components=window_size, n_iter=5, random_state=None)

except ImportError:
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]
querys = ["you", "year", "car", "toyota"]
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)


