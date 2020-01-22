import numpy as np


def lcp(topic_words, word_frequency, word_doc_frequency):
    n_top = len(topic_words)

    metric_value = 0
    for j in range(1, n_top):
        for i in range(0, j):
            word_i = topic_words[i]
            word_j = topic_words[j]

            wi = word_frequency[word_i]
            wi_and_wj = len(word_doc_frequency[word_i].intersection(
                word_doc_frequency[word_j]))

            div = wi_and_wj / wi

            if wi_and_wj != 0:
                metric_value += np.log(div)
            else:
                metric_value += 0

    return metric_value
