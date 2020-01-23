import numpy as np


def coherence(topic_words, word_frequency, word_frequency_in_documents, smoothing=1.0):
    """Coherence topic quality metric for a topic.

    Calculates the Coherence topic quality metric for one individual topic based on the topic words.

    Args:
        topic_words: list
            Words that compose one individual topic.
        word_frequency: dict
            Frequency of each word in corpus.
        word_frequency_in_documents: dict
            Frequency of each word for each document in corpus.
        smoothing: float, default=1.0
            Smoothing value for the coherence metric.

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

            word_frequency_1 = word_frequency_in_documents[word_1]
            word_frequency_2 = word_frequency_in_documents[word_2]

            count_wi = word_frequency[word_2]
            count_wi_wj = float(len(word_frequency_2.intersection(word_frequency_1)))
            metric_value += np.log10((count_wi_wj + smoothing) / count_wi)

    return metric_value
