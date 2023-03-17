from word_embedding.helper import clean_and_tokenize, vocabulary

def test_vocabulary():
    corpus = ['I walked down down the boulevard',
          'I walked down the avenue',
          'I ran down the boulevard',
          'I walk down the city',
          'I walk down the the avenue']
    voc, counts = vocabulary(corpus, voc_threshold = 3)
    assert voc == {'down': 0, 'the': 1, 'i': 2, 'UNK': 3}
    assert counts == {'down': 6, 'the': 6, 'i': 5, 'UNK': 1}

    voc, counts = vocabulary(corpus)
    assert voc == {'down': 0, 'the': 1, 'i': 2, 'walked': 3, 'boulevard': 4, 'avenue': 5, 'walk': 6, 'ran': 7, 'city': 8, 'UNK': 9}
    assert counts == {'down': 6, 'the': 6, 'i': 5, 'walked': 2, 'boulevard': 2, 'avenue': 2, 'walk': 2, 'ran': 1, 'city': 1, 'UNK': 1}