import re
import os
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    Source: https://stackoverflow.com/questions/34968722/softmax-function-python
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preprocess(text, chars='', remove_all_special=True, expand=True, split_numbers=True):
    """
    function that removes whitespaces, converts to lowercase, etc.
    :param split_numbers: split 45787 to 4 5 7 8 7
    :param remove_all_special: remove all characters but  alpha-numerical, spaces, hyphens, quotes
    :param expand: expand 'll, 'm and similar expressions to reduce the number of different tokens
    :param text: text input
    :param chars: chars to be removed
    :return: cleaned up text
    """

    # fix bad newlines (replace with spaces), unify quotes
    text = text.replace('\\n', ' ').replace('‘', '\'').replace('’', '\'').replace('”', '"').replace('“', '"')

    # optionally remove all given characters
    for c in chars:
        if c in text:
            text = text.replace(c, '')

    # convert to lowercase
    text = text.lower()

    # remove all characters except alphanum, spaces and - ' "
    if remove_all_special:
        text = re.sub('[^ \-\sA-Za-z0-9"\']+', ' ', text)

    # split numbers into digits to avoid infinite vocabulary size if random numbers are present:
    if split_numbers:
        text = re.sub('[0-9]', ' \g<0> ', text)

    # expand unambiguous 'm, 't, 're, ... expressions
    if expand:
        text = text. \
            replace('\'m ', ' am '). \
            replace('\'re ', ' are '). \
            replace('won\'t', 'will not'). \
            replace('n\'t', ' not'). \
            replace('\'ll ', ' will '). \
            replace('\'ve ', ' have '). \
            replace('\'s', ' \'s')

    return text


def load_embeddings(path):
    """
    loads embeddings from a file and their their index (a dictionary of words with coefficients)
    tested on GloVe
    :param path: path to the embedding file
    :return:
    """
    embeddings_index = {}
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefficients
    f.close()

    logger.info('Imported embeddings are using %s word vectors.' % len(embeddings_index))
    return embeddings_index
