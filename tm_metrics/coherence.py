import numpy as np

from .feature_extraction.text import get_word_frequencies


def coherence(topic, documents, smoothing=1.0):
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
    top_words = topic.split()
    n_top_words = len(top_words)
    metric_value = 0.0

    word_frequency, word_frequency_in_documents = get_word_frequencies(documents)

    for m in range(1, n_top_words):
        for l in range(0, m):
            word_1 = top_words[m]
            word_2 = top_words[l]

            word_frequency_1 = word_frequency_in_documents[word_1]
            word_frequency_2 = word_frequency_in_documents[word_2]

            count_wi = word_frequency[word_2]
            count_wi_wj = float(len(word_frequency_2.intersection(word_frequency_1)))
            metric_value += np.log10((count_wi_wj + smoothing) / count_wi)

    return metric_value
