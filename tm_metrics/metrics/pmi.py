import numpy as np


def pmi(topic_words, word_frequency, word_frequency_in_documents, n_docs, normalise=False):
    """PMI/NPMI topic quality metric for a topic.

    Calculates the PMI/NPMI topic quality metric for one individual topic based on the topic words.

    Args:
        topic_words: list
            Words that compose one individual topic.
        word_frequency: dict
            Frequency of each word in corpus.
        word_frequency_in_documents: dict
            Frequency of each word for each document in corpus.
        n_docs: int
            Number of documents in the corpus.
        normalise: bool, default=False
            Where to normalise (NPMI) or not (PMI).

    Returns:
        pmi: float
            Resultant PMI metric value for the topic.
        npmi: float
            Resultant NPMI metric value for the topic.
    """
    n_top = len(topic_words)
    pmi = 0.0
    npmi = 0.0

    for j in range(1, n_top):
        for i in range(0, j):
            ti = topic_words[i]
            tj = topic_words[j]

            c_i = word_frequency[ti]
            c_j = word_frequency[tj]
            c_i_and_j = len(word_frequency_in_documents[ti].intersection(word_frequency_in_documents[tj]))

            dividend = (c_i_and_j + 1.0) / float(n_docs)
            divisor = ((c_i * c_j) / float(n_docs) ** 2)
            pmi += np.log(dividend / divisor)

            npmi += -1.0 * np.log((c_i_and_j + 0.01) / float(n_docs))

    npmi = pmi / npmi

    if normalise:
        return npmi
    else:
        return pmi
