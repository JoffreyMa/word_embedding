# Packages nécessaires
import numpy as np
import re

def clean_and_tokenize(text):
    """
    Cleaning a document with:
        - Lowercase        
        - Removing anything that is not letters
    (Of course, we can use regular expression to do more elaborate pre-processing....)
    And separate the document into words by simply splitting at spaces
    Params:
        text (string): a sentence or a document
    Returns:
        tokens (list of strings): the list of tokens (word units) forming the document
    """        
    # Lowercase
    text = text.lower()
    # Remove anything but letters
    text = re.sub(r"[^a-z]+", " ", text)
    
    tokens = text.split()        
    return tokens

def vocabulary(corpus, voc_threshold=None):
    """
    Function using word counts to build a vocabulary
    Params:
        corpus (list of list of strings): corpus of sentences
        voc_threshold (int): maximum size of the vocabulary - 0 (default) indicates there is no max
    Returns:
        vocabulary (dictionary): keys: list of distinct words across the corpus
                                 values: indexes corresponding to each word sorted by frequency
        vocabulary_word_counts (dictionary): keys: list of distinct words across the corpus
                                             values: word counts in the corpus
    """
    word_counts = {}

    # Count the occurrences of each word in the corpus
    for sentence in corpus:
        for word in clean_and_tokenize(sentence):
            word_counts[word] = word_counts.get(word, 0) + 1

    # Sort the words based on their frequency
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # Truncate the sorted word counts based on the vocabulary threshold if provided
    if voc_threshold and voc_threshold > 0:
        sorted_word_counts = sorted_word_counts[:voc_threshold]
    sorted_word_counts.append(('UNK', 1))

    # Create the vocabulary and vocabulary_word_counts dictionaries
    vocabulary = {word: index for index, (word, _) in enumerate(sorted_word_counts)}
    vocabulary_word_counts = {word: count for word, count in sorted_word_counts}


    return vocabulary, vocabulary_word_counts

def co_occurence_matrix(corpus, vocabulary, window=0, distance_weighting=False):
    """
    Params:
        corpus (list of list of strings): corpus of sentences
        vocabulary (dictionary): words to use in the matrix
        window (int): size of the context window; when 0, the context is the whole sentence
        distance_weighting (bool): indicates if we use a weight depending on the distance between words for co-oc counts
    Returns:
        matrix (array of size (len(vocabulary), len(vocabulary))): the co-oc matrix, using the same ordering as the vocabulary given in input    
    """ 
    l = len(vocabulary)
    M = np.zeros((l, l))
    for sent in corpus:
        # Get the sentence:
        sent = clean_and_tokenize(sent)
        # Get the indexes of the sentence using the vocabulary: 
        sent_idx = [vocabulary.get(word, -1) for word in sent]
        
        # Go through the sentence indexes and add 1 / dist(i, j) to M[i, j] if words with index i and j appear in the same window. 
        for i, idx_i in enumerate(sent_idx):
            # Check if the word is recognized by the vocabulary:
            if idx_i > -1:
                # If we consider a limited context:
                if window > 0:
                    l_ctx_idx = sent_idx[max(0, i - window):i]
                # If we consider that the context is the entire sentence:
                else:
                    l_ctx_idx = sent_idx[:i]
                    
                # Go through this list and update M[i, j]:    
                for j, idx_j in enumerate(l_ctx_idx):
                    # ... making sure that the word corresponding to 'idx_j' is recognized by the vocabulary
                    if idx_j > -1:
                        # Calculate the weight:
                        if distance_weighting:
                            weight = 1.0 / (i - j)
                        else:
                            weight = 1.0
                        M[idx_i, idx_j] += weight
                        M[idx_j, idx_i] += weight
    return M
