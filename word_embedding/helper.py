# Contrôle de version
import sys

# Packages nécessaires
import numpy as np
import os
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
