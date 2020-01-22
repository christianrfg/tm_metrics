import numpy as np


def pmi(topic_words, word_frequency, word_doc_frequency, n_docs, normalise=False):
    n_top = len(topic_words)
    pmi = 0.0
    npmi = 0.0

    for j in range(1, n_top):
        for i in range(0, j):
            ti = topic_words[i]
            tj = topic_words[j]

            c_i = word_frequency[ti]
            c_j = word_frequency[tj]
            c_i_and_j = len(word_doc_frequency[ti].intersection(word_doc_frequency[tj]))

            dividend = (c_i_and_j + 1.0) / float(n_docs)
            divisor = ((c_i * c_j) / float(n_docs) ** 2)
            pmi += np.log(dividend / divisor)

            npmi += -1.0 * np.log((c_i_and_j + 0.01) / float(n_docs))

    npmi = pmi / npmi

    if normalise:
        return npmi
    else:
        return pmi
