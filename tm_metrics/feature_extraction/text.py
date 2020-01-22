from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_tfidf_matrices(documents):
    """Generates TF-IDF matrices.

    Generates the TF-IDF representation based on the Scikit-Learn implementation. We generated the default
    TF-IDF matrix and the transposed, that are both used in the topic quality metrics.

    Args:
        documents: type, default="x"
            Small comment...

    Returns:
        Large comment...
    """
    tfidf_vec_model = TfidfVectorizer()
    tfidf_matrix = tfidf_vec_model.fit_transform(documents)
    tfidf_matrix_transpose = tfidf_matrix.transpose()

    return tfidf_matrix, tfidf_matrix_transpose


def get_vocabulary(documents):
    """
    Description
    -----------
    TODO

    Parameters
    -----------
    :param documents: list
        TODO

    Return
    -----------
    :return vocabulary: list
        TODO

    Example
    -----------
    TODO
    """
    cv_model = CountVectorizer(binary=True)
    cv_model.fit(documents)

    vocabulary = cv_model.get_feature_names()
    vocabulary = list(map(str, vocabulary))

    return vocabulary


def get_word_frequencies(documents):
    """
    Description
    -----------
    TODO

    Parameters
    -----------
    :param documents: list
        TODO

    Returns
    -----------
    :returns:

    Example
    -----------
    TODO
    """
    cv_model = CountVectorizer(binary=True)
    tf_matrix = cv_model.fit_transform(documents)
    tf_matrix_transpose = tf_matrix.transpose()

    vocabulary = get_vocabulary(documents)
    n_words = len(vocabulary)

    word_frequency = {}
    word_frequency_in_documents = {}

    for word_idx in range(n_words):
        word = vocabulary[word_idx]
        tf_word = tf_matrix_transpose[word_idx]

        # getnnz -> Get the count of explicitly-stored values (nonzeros)
        word_frequency[word] = float(tf_word.getnnz(1))
        # nonzero -> Return the indices of the elements that are non-zero
        word_frequency_in_documents[word] = set(tf_word.nonzero()[1])

    return word_frequency, word_frequency_in_documents
