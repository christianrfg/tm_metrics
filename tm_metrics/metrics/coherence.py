import numpy as np


def coherence(topic_words, word_frequency, word_frequency_in_documents, smoothing=1.0):
    """
    Description
    -----------
    TODO

    Parameters
    -----------
    :param topic: str
        TODO
    :param documents: list
        TODO
    :param smoothing: float
        TODO

    Return
    -----------
    :return metric_value: float
        TODO

    Example
    -----------
    TODO
    """
    n_top_words = len(topic_words)
    metric_value = 0.0

    for w1 in range(1, n_top_words):
        for w2 in range(0, w1):
            word_1 = topic_words[w1]
            word_2 = topic_words[w2]

            word_frequency_1 = word_frequency_in_documents[word_1]
            word_frequency_2 = word_frequency_in_documents[word_2]

            count_wi = word_frequency[word_2]
            count_wi_wj = float(len(word_frequency_2.intersection(word_frequency_1)))
            metric_value += np.log10((count_wi_wj + smoothing) / count_wi)

    return metric_value
