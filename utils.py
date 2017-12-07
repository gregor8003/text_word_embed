import itertools
import os
import pickle
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk


MDSD_MAIN_PATH = 'mdsd'

MDSD_DOMAIN_PATHS = (
    'books',
    'dvd',
    'electronics',
    'kitchen_&_housewares',
)

MDSD_DOMAIN_FILES = (
    'negative.review',
    'positive.review',
    'unlabeled.review'
)

MDSD_WORDFREQ_JSON_FILE = 'wordfreq.json'
MDSD_WORDFREQ_CSV_FILE = 'wordfreq.csv'
MDSD_WORD2INDEX_PCK_FILE = 'word2index.pck'
MDSD_INDEX2WORD_PCK_FILE = 'index2word.pck'

MDSD_CSV_FILE = 'mdsd.csv'

MDSD_CSV_INDEXED_FILE = 'mdsd.indexed.csv'

MDSD_CSV_CBOW_DATA_INPUT_FILE = 'mdsd.csv.cbow.data.input.{:06d}'

MDSD_CSV_CBOW_DATA_INPUT_FILE_RE = (
    re.compile('mdsd\.csv\.cbow\.data\.input\.[0-9]{6}\.npy')
)

MDSD_CSV_CBOW_DATA_OUTPUT_FILE = 'mdsd.csv.cbow.data.output.{:06d}'

MDSD_CSV_CBOW_DATA_OUTPUT_FILE_RE = (
    re.compile('mdsd\.csv\.cbow\.data\.output\.[0-9]{6}\.npy')
)

MDSD_CBOW_EMBEDDING_WEIGHTS_FILE = 'mdsd.cbow.embedding.weights.npy'

MDSD_CBOW_WORDVEC_CLOSEST_NEIGHBORS_FILE = (
    'mdsd.cbow.wordvec.closest.neighbors.npy'
)

MDSD_CBOW_WORDVEC_CLOSEST_NEIGHBORS_CSV_FILE = (
    'mdsd.cbow.wordvec.closest.neighbors.csv'
)


def _get_domain_paths(main_path, domain_paths, domain_files, exclusions=()):
    paths = []
    for domain_path in domain_paths:
        for domain_file in domain_files:
            path = os.path.join(main_path, domain_path, domain_file)
            # get chars until first .
            domain_file_part = domain_file[:domain_file.find('.')]
            if path not in exclusions:
                paths.append((path, domain_path, domain_file_part))
    return paths


def get_mdsd_domain_paths(main_path=MDSD_MAIN_PATH,
                          domain_paths=MDSD_DOMAIN_PATHS,
                          domain_files=MDSD_DOMAIN_FILES,
                          exclusions=()):
    return _get_domain_paths(main_path, domain_paths, domain_files, exclusions)


def _get_file_path(main_path, file_path):
    return os.path.join(main_path, file_path)


def get_mdsd_wordfreq_json_file(main_path=MDSD_MAIN_PATH,
                                file_path=MDSD_WORDFREQ_JSON_FILE):
    return _get_file_path(main_path, file_path)


def get_mdsd_wordfreq_csv_file(main_path=MDSD_MAIN_PATH,
                               file_path=MDSD_WORDFREQ_CSV_FILE):
    return _get_file_path(main_path, file_path)


def get_mdsd_word2index_pck_file(main_path=MDSD_MAIN_PATH,
                                 file_path=MDSD_WORD2INDEX_PCK_FILE):
    return _get_file_path(main_path, file_path)


def get_mdsd_index2word_pck_file(main_path=MDSD_MAIN_PATH,
                                 file_path=MDSD_INDEX2WORD_PCK_FILE):
    return _get_file_path(main_path, file_path)


def read_mdsd_word2index_pck_file(main_path=MDSD_MAIN_PATH):
    path = get_mdsd_word2index_pck_file(main_path=main_path)
    with open(path, 'rb') as f:
        word2index = pickle.load(f)
    return word2index


def read_mdsd_index2word_pck_file(main_path=MDSD_MAIN_PATH):
    path = get_mdsd_index2word_pck_file(main_path=main_path)
    with open(path, 'rb') as f:
        index2word = pickle.load(f)
    return index2word


def get_vocabulary_size(main_path=MDSD_MAIN_PATH):
    index2word = read_mdsd_index2word_pck_file(main_path=main_path)
    max_index = max(index2word, key=int)
    return max_index


def get_mdsd_csv_file(main_path=MDSD_MAIN_PATH, file_path=MDSD_CSV_FILE):
    return _get_file_path(main_path, file_path)


def get_mdsd_csv_indexed_file(main_path=MDSD_MAIN_PATH,
                              file_path=MDSD_CSV_INDEXED_FILE):
    return _get_file_path(main_path, file_path)


def get_mdsd_csv_cbow_data_input_file(main_path=MDSD_MAIN_PATH,
                                      file_path=MDSD_CSV_CBOW_DATA_INPUT_FILE):
    return _get_file_path(main_path, file_path)


def get_mdsd_csv_cbow_data_output_file(
        main_path=MDSD_MAIN_PATH, file_path=MDSD_CSV_CBOW_DATA_OUTPUT_FILE):
    return _get_file_path(main_path, file_path)


def get_mdsd_cbow_embedding_weights_file(
        main_path=MDSD_MAIN_PATH, file_path=MDSD_CBOW_EMBEDDING_WEIGHTS_FILE):
    return _get_file_path(main_path, file_path)


def get_mdsd_cbow_wordvec_closest_neighbors_file(
        main_path=MDSD_MAIN_PATH,
        file_path=MDSD_CBOW_WORDVEC_CLOSEST_NEIGHBORS_FILE):
    return _get_file_path(main_path, file_path)


def get_mdsd_cbow_wordvec_closest_neighbors_csv_file(
        main_path=MDSD_MAIN_PATH,
        file_path=MDSD_CBOW_WORDVEC_CLOSEST_NEIGHBORS_CSV_FILE):
    return _get_file_path(main_path, file_path)


def _get_matched_paths(main_path, file_path_re):
    listing = os.listdir(main_path)
    filenames = sorted([f for f in listing if file_path_re.match(f)])
    paths = [os.path.join(main_path, f) for f in filenames]
    return paths


def get_mdsd_csv_cbow_data_input_files_iter(
        main_path=MDSD_MAIN_PATH,
        file_path_re=MDSD_CSV_CBOW_DATA_INPUT_FILE_RE):
    return _get_matched_paths(main_path, file_path_re)


def get_mdsd_csv_cbow_data_output_files_iter(
        main_path=MDSD_MAIN_PATH,
        file_path_re=MDSD_CSV_CBOW_DATA_OUTPUT_FILE_RE):
    return _get_matched_paths(main_path, file_path_re)


DEFAULT_PUNCTS = string.punctuation + string.digits


def default_wordfreq_accept_predicate(word, punctuations=DEFAULT_PUNCTS):
    return not any(c in word for c in punctuations)


def default_wordindex_accept_predicate(word, frequency):
    return True


PARTIALS = {
    '\'s': (),
    '\'m': ('am',),
    'n\'t': ('not',),
    '\'ve': ('have',),
    '\'ll': ('will'),
    'can\'t': ('can', 'not',),
    'cannot': ('can', 'not',),
    'should\'ve': ('should', 'have',),
}


def process_partials(word):
    return PARTIALS.get(word, (word,))


STOPWORDS = set(stopwords.words('english'))

STEMMER = PorterStemmer()


def process_raw_word(word):
    return STEMMER.stem(word)


# !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
# no '
PUNCTUATION = r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~""" + string.digits

PUNCT_TRANSTABLE = str.maketrans('', '', PUNCTUATION)


def remove_punctuation(word, trans_table=PUNCT_TRANSTABLE):
    return word.translate(trans_table)


class SentenceTokenizer(object):

    def process(self, document):
        for sentence in nltk.sent_tokenize(document):
            for word in nltk.word_tokenize(sentence):
                for p in process_partials(word):
                    w = process_raw_word(remove_punctuation(p))
                    ws = frozenset((w,))
                    if not ws & STOPWORDS:
                        if len(w) > 0 and w != "''":
                            yield w

    def __call__(self, document):
        yield from self.process(document)


INDEX_UNKNOWN_WORD = 0
WORD_UNKNOWN_WORD = '<Unknown Word>'


UNKNOWN = 0
POSITIVE = 1
NEGATIVE = 2
UNLABELED = 3

FIELD_DOMAIN = 'Domain'
FIELD_TEXT = 'Original Text'
FIELD_PREPROCESSED = 'Preprocessed Text'
FIELD_RATING = 'Rating'
FIELD_LABEL = 'Label'
FIELD_INDEXED = 'Indexed Text'

CSV_FIELD_NAMES = [
    FIELD_TEXT, FIELD_PREPROCESSED, FIELD_DOMAIN, FIELD_RATING, FIELD_LABEL
]

CSV_INDEXED_FIELD_NAMES = [
    FIELD_INDEXED, FIELD_DOMAIN, FIELD_RATING, FIELD_LABEL
]


def get_label(domain):
    if domain.startswith('positive'):
        return POSITIVE
    elif domain.startswith('negative'):
        return NEGATIVE
    elif domain.startswith('unlabeled'):
        return UNLABELED
    return UNKNOWN


def get_label_from_rating(rating):
    if rating in (4, 5):
        return POSITIVE
    elif rating in (1, 2):
        return NEGATIVE
    else:
        return None


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)
