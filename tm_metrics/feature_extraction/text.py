from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_tfidf_matrices(documents):
    """Generates TF-IDF matrices.

    Generates the TF-IDF representation based on the Scikit-Learn implementation. We generated the default
    TF-IDF matrix and the transposed, that are both used in the topic quality metrics.

    Args:
        documents: list
            List where each element is a entire document.

    Returns:
        tfidf_matrix: sparse matrix
            TF-IDF matrix.
        tfidf_matrix_transpose: sparse matrix
            TF-IDF matrix transposed.
    """
    tfidf_vec_model = TfidfVectorizer()
    tfidf_matrix = tfidf_vec_model.fit_transform(documents)
    tfidf_matrix_transpose = tfidf_matrix.transpose()

    return tfidf_matrix, tfidf_matrix_transpose


def get_vocabulary(documents):
    """Generates the corpus vocabulary.

    Generates the corpus vocabulary from the CountVectorizer of Scikit-Learn.

    Args:
        documents: list
            List where each element is a entire document.

    Returns:
        vocabulary: list
            List of words present in the corpus.
    """
    cv_model = CountVectorizer(binary=True)
    cv_model.fit(documents)

    vocabulary = cv_model.get_feature_names()
    vocabulary = list(map(str, vocabulary))

    return vocabulary


def get_word_frequencies(documents):
    """Word frequencies in documents.

    Count frequency of words and frequency of words in documents.

    Args:
        documents: list
            List where each element is a entire document.

    Returns:
        word_frequency: dict
            Frequency of each word in corpus.
        word_frequency_in_documens: dict
            Frequency of each word for each document in corpus.
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
