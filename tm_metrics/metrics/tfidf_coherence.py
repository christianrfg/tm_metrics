import numpy as np


def tfidf_coherence(topic_words, tfidf_matrix_transpose, vocabulary, smoothing=1.0):
    """TFIDF-Coherence topic quality metric for a topic.

    Calculates the TFIDF-Coherence topic quality metric for one individual topic based on the topic words.

    Args:
        topic_words: list
            Words that compose one individual topic.
        tfidf_matrix_transpose: dict
            Frequency of each word in corpus.
        vocabulary: dict
            Frequency of each word for each document in corpus.
        smoothing: float, default=1.0
            Smoothing value for the TFIDF-Coherence metric.

    Returns:
        metric_value: float
            Resultant Coherence metric value for the topic.
    """
    n_top_words = len(topic_words)
    metric_value = 0.0

    for w1 in range(1, n_top_words):
        for w2 in range(0, w1):
            word_1 = topic_words[w1]
            word_2 = topic_words[w2]

            wi_index = vocabulary.index(word_1)
            wj_index = vocabulary.index(word_2)

            wi = tfidf_matrix_transpose[wi_index]
            wj = tfidf_matrix_transpose[wj_index]

            sum_tfidf_wi = sum(wi.data)

            docs_with_wi = set(wi.nonzero()[1])
            docs_with_wj = set(wj.nonzero()[1])
            docs_with_wi_and_wj = docs_with_wi.intersection(docs_with_wj)

            sum_w1_w2 = 0.0
            for k in docs_with_wi_and_wj:
                wi_data = tfidf_matrix_transpose.getrow(wi_index).getcol(
                    k).data[0]
                wj_data = tfidf_matrix_transpose.getrow(wj_index).getcol(
                    k).data[0]

                sum_w1_w2 += wi_data * wj_data

            metric_value += np.log10((sum_w1_w2 + smoothing) / sum_tfidf_wi)

    return metric_value
