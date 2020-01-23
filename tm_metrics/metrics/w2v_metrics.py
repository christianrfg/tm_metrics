import scipy.spatial.distance as sci_dist


def topic_w2v(topic_words, word_embedding):
    """Word Embedding topic quality metric for a topic.

    Calculates the Cosine Distance, L1 Distance, L2 Distance and Coordinate Distance topic quality metrics
    for one individual topic based on the topic words.

    Args:
        topic_words: list
            Words that compose one individual topic.
        word_embedding: gensim.KeyedVectors
            Mapping between words and vectors for the Word2Vec model. Generate with Gensim.

    Returns:
        cosine_distance: float
            Resultant Cosine Distance for the topic.
        l1_distance: float
            Resultant L1 Distance metric for the topic.
        l2_distance: float
            Resultant L2 Distance metric for the topic.
        coordinate_distance: float
            Resultant Coordinate Distance metric for the topic.
    """
    cosine_distance = 0.0
    l1_distance = 0.0
    l2_distance = 0.0
    coordinate_distance = 0.0

    n_top = len(topic_words)

    t = float(n_top)
    t = t * (t - 1.0)

    for word_i_idx in range(n_top):
        for word_j_idx in range(word_i_idx + 1, n_top):
            try:
                word_i = word_embedding[topic_words[word_i_idx]]
                word_j = word_embedding[topic_words[word_j_idx]]
            except KeyError:
                continue

            cosine_distance += (sci_dist.cosine(word_i, word_j))
            l1_distance += (sci_dist.euclidean(word_i, word_j))
            l2_distance += (sci_dist.sqeuclidean(word_i, word_j))
            coordinate_distance += (sci_dist.sqeuclidean(word_i, word_j))

    cosine_distance = cosine_distance / t
    l1_distance = l1_distance / t
    l2_distance = l2_distance / t
    coordinate_distance = coordinate_distance / t

    return cosine_distance, l1_distance, l2_distance, coordinate_distance
