import numpy as np


def lcp(topic_words, word_frequency, word_frequency_in_documents):
    """LCP topic quality metric for a topic.

    Calculates the LCP topic quality metric for one individual topic based on the topic words.

    Args:
        topic_words: list
            Words that compose one individual topic.
        word_frequency: dict
            Frequency of each word in corpus.
        word_frequency_in_documents: dict
            Frequency of each word for each document in corpus.

    Returns:
        metric_value: float
            Resultant LCP metric value for the topic.
    """
    n_top = len(topic_words)

    metric_value = 0
    for j in range(1, n_top):
        for i in range(0, j):
            word_i = topic_words[i]
            word_j = topic_words[j]

            wi = word_frequency[word_i]
            wi_and_wj = len(word_frequency_in_documents[word_i].intersection(
                word_frequency_in_documents[word_j]))

            div = wi_and_wj / wi

            if wi_and_wj != 0:
                metric_value += np.log(div)
            else:
                metric_value += 0

    return metric_value
