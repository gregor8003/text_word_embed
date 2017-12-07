import collections
import csv
import json
import operator
import pickle
import sys

import pandas as pd

from utils import (
    get_mdsd_csv_file,
    get_mdsd_csv_indexed_file,
    get_mdsd_wordfreq_json_file,
    get_mdsd_wordfreq_csv_file,
    get_mdsd_word2index_pck_file,
    get_mdsd_index2word_pck_file,
    default_wordindex_accept_predicate,
    CSV_INDEXED_FIELD_NAMES,
    INDEX_UNKNOWN_WORD,
    FIELD_PREPROCESSED,
    FIELD_INDEXED,
    FIELD_DOMAIN,
    FIELD_RATING,
    FIELD_LABEL,
)


def build_mdsd_word_frequencies(df):
    word_freqs = collections.defaultdict(int)
    # get proper column index by column name
    col_index = df.columns.get_loc(FIELD_PREPROCESSED)
    for rowtuple in df.itertuples(index=False, name=None):
        preprocessed_text = rowtuple[col_index]
        for word in preprocessed_text.split():
            word_freqs[word] += 1
    return word_freqs


def build_mdsd_word_indexes(
        word_frequencies, accept_predicate=default_wordindex_accept_predicate):
    word2index = {}
    index2word = {}
    for index, (word, freq) in enumerate(word_frequencies, 1):
        if accept_predicate(word, freq):
            word2index[word] = index
            index2word[index] = word
    return word2index, index2word


def index_texts(df, csv_writer, word2index):
    for rowtuple in df.itertuples(index=False, name=None):
        _, preprocessed_text, domain, rating, label = rowtuple
        indexed_text = []
        for word in preprocessed_text.split():
            word_index = INDEX_UNKNOWN_WORD
            try:
                word_index = word2index[word]
            except KeyError:
                pass
            indexed_text.append(word_index)
        row_content = {
            FIELD_INDEXED: ' '.join([str(s) for s in indexed_text]),
            FIELD_DOMAIN: domain,
            FIELD_RATING: rating,
            FIELD_LABEL: label,
        }
        csv_writer.writerow(row_content)


def main():

    main_path = sys.argv[1]

    mdsd_csv_file = get_mdsd_csv_file(main_path=main_path)

    mdsd_csv_indexed_file = get_mdsd_csv_indexed_file(main_path=main_path)

    print('Reading', mdsd_csv_file, '... ', end='', flush=True)
    df = pd.read_csv(mdsd_csv_file, encoding='utf-8')
    # remove NaNs
    df = df[pd.notnull(df[FIELD_PREPROCESSED])]
    print('done')

    wordfreq_json_file = get_mdsd_wordfreq_json_file()
    wordfreq_csv_file = get_mdsd_wordfreq_csv_file()
    word2index_pck_file = get_mdsd_word2index_pck_file()
    index2word_pck_file = get_mdsd_index2word_pck_file()

    print('Calculating word frequencies ... ', end='', flush=True)
    word_freqs = build_mdsd_word_frequencies(df)
    print('done')

    print('Writing', wordfreq_json_file, '... ', end='', flush=True)
    with open(wordfreq_json_file, 'w') as wf:
        json.dump(word_freqs, wf, indent=2)
    print('done')

    print('Writing', wordfreq_csv_file, '... ', end='', flush=True)
    sorted_by_freqs = sorted(
        word_freqs.items(), key=operator.itemgetter(1), reverse=True
    )
    with open(wordfreq_csv_file, 'wt', encoding='utf-8') as wf:
        csvw = csv.writer(wf)
        csvw.writerow(('Word', 'Frequency'))
        for word, freq in sorted_by_freqs:
            csvw.writerow((word, str(freq)))
    print('done')

    print('Building word indices', '... ', end='', flush=True)
    word2index, index2word = build_mdsd_word_indexes(sorted_by_freqs)
    print('done')

    print('Writing', word2index_pck_file, '... ', end='', flush=True)
    with open(word2index_pck_file, 'wb') as wf:
        pickle.dump(word2index, wf, protocol=0)
    print('done')

    print('Writing', index2word_pck_file, '... ', end='', flush=True)
    with open(index2word_pck_file, 'wb') as wf:
        pickle.dump(index2word, wf, protocol=0)
    print('done')

    print('Indexing texts', '... ', end='', flush=True)
    with open(mdsd_csv_indexed_file,
              mode='wt', encoding='utf-8', newline='') as wf:
        csvf = csv.DictWriter(wf, CSV_INDEXED_FIELD_NAMES, dialect='excel')
        csvf.writeheader()
        index_texts(df, csvf, word2index)
    print('done')

    print('All done')


if __name__ == '__main__':
    main()
