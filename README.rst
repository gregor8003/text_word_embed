text_word_embed
===============

This repository contains example (and very preliminary) implementation of
word embeddings, highly popular concept used in text processing. Essentially,
a word is transformed into a vector in high-dimensional Euclidean space. They
are particularly effective when used with quantitative techniques of text
processing, e.g. deep neural networks. The embeddings produced here are of
poor quality, but one can see the process of word clustering being initiated.


Installation
------------

- Install latest `Anaconda3 <https://www.anaconda.com/download>`_

- Download and unpack latest text_word_embed source distribution, or simply
  clone the repo

- Create conda environment

.. code-block:: bash

    $ cd text_word_embed-<x.y.z.www>
    $ conda env create -f environment.yml
    $ activate text_word_embed

- See `Dataset(s)`_ for preparing data

- See `Usage`_ for scripts

Dataset(s)
----------

NOTE: dataset(s) are not included and must be downloaded separately.

MDSD
^^^^

Multi-Domain Sentiment
Dataset `(MDSD) version 1 <https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html>`
contains Amazon reviews from four categories: books, dvd, electronics,
kitchen & housewares. They are rated from 1 to 5, where 1 or 2 means "negative",
and 4 or 5 means "positive". The dataset contains labeled and unlabeled data.

* Download `domain_sentiment_data.tar.gz <https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz>`
and unpack to the directory of choice.

* Move/copy four subdirectories with categories data to directory named 'mdsd',
or simply rename unpacked one.

* Download `book.unlabeled.gz <https://www.cs.jhu.edu/~mdredze/datasets/sentiment/book.unlabeled.gz>`,
unpack it and place as 'mdsd/books/unlabeled.review'.

You should end up with the following directory structure:

::
    +---mdsd
        |
        +---books
        |     negative.review
        |     positive.review
        |     unlabeled.review
        +---dvd
        |     negative.review
        |     positive.review
        |     unlabeled.review
        +---electronics
        |     negative.review
        |     positive.review
        |     unlabeled.review
        +---kitchen_&_housewares
              negative.review
              positive.review
              unlabeled.review

Usage
-----

mdsd_to_csv
^^^^^^^^^^^

.. code-block:: bash

    $ python3 -m mdsd_to_csv [path_to_mdsd_dir]

The script ``mdsd_to_csv`` reads data files from ``mdsd`` directory (with
structure described above), and produces CSV file ``mdsd.csv`` in ``mdsd``
directory.

After the script finishes successfully, you should end up with:

::
    +---mdsd
        |  mdsd.csv
        +---books
        ..


build_vocabulary_index_texts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ python3 -m build_vocabulary_index_texts [path_to_mdsd_dir]

The script ``build_vocabulary_index_texts`` reads ``mdsd.csv`` file from
``mdsd`` directory, calculates word frequencies across the whole dataset,
builds vocabulary, and encodes texts into numerical representation.

The script produces the following new files in ``mdsd`` directory:

* ``wordfreq.json``, ``wordfreq.csv`` - contain word frequencies, in descending
order

* ``word2index.pck`` - pickled dictionary with mapping ``word -> index``

* ``index2word.pck`` - pickled dictionary with mapping ``index -> word``

* ``mdsd.indexed.csv`` - CSV file similar to ``mdsd.csv``, but contains text
in indexed form, that is, every text document is transformed into sequence of
word indexes in vocabulary

After the script finishes successfully, the files  in ``mdsd`` directory are:

::
    +---mdsd
        |  mdsd.csv
        |  wordfreq.json
        |  wordfreq.csv
        |  word2index.pck
        |  index2word.pck
        |  mdsd.indexed.csv
        +---books
        ..


generate_cbow_data
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ python3 -m generate_cbow_data [path_to_mdsd_dir] [half_window_size] [negative_examples_cnt]

The script ``generate_cbow_data`` reads ``mdsd.indexed.csv`` file from
``mdsd`` directory (created by ``build_vocabulary_index_texts`` script), and
generates training data for neural network.

The default values for parameters are:

* ``half_window_size`` - ``3``

* ``negative_examples_cnt`` - ``30``

See `README.data.rst </README.data.rst>`_ for details about training data.

The script produces the following sequence of files in ``mdsd`` directory:

* ``mdsd.csv.cbow.data.input.NNNNNN.npy``, where NNNNNN is integer
* ``mdsd.csv.cbow.data.output.NNNNNN.npy``, where NNNNNN is integer

After the script finishes successfully, the files  in ``mdsd`` directory are:

::
    +---mdsd
        |  mdsd.csv
        |  wordfreq.json
        |  wordfreq.csv
        |  word2index.pck
        |  index2word.pck
        |  mdsd.indexed.csv
        |  mdsd.csv.cbow.data.input.000001.npy
        |  mdsd.csv.cbow.data.input.000002.npy
        |  ..
        |  mdsd.csv.cbow.data.input.NNNNNN.npy
        |  mdsd.csv.cbow.data.output.000001.npy
        |  mdsd.csv.cbow.data.output.000002.npy
        |  ..
        |  mdsd.csv.cbow.data.output.NNNNNN.npy
        +---books
        ..


train_cbow
^^^^^^^^^^

.. code-block:: bash

    $ python3 -m train_cbow [path_to_mdsd_dir] [embedding_size] [half_window_size] [negative_examples_cnt]

The script ``train_cbow`` reads sequence of files
``mdsd.csv.cbow.data.input.NNNNNN.npy``/``mdsd.csv.cbow.data.output.NNNNNN.npy``
from ``mdsd`` directory (created by ``generate_cbow_data`` script), and trains
neural network with supplied input/output data. During this training, word
embeddings are learned as byproduct.

The parameter ``embedding_size`` is a size of embedding vector learned for
every word in the vocabulary. For instance, if there are 1000 words in vocabulary,
and embedding size is 300, then 1000 vectors of size 300 will be learned, and
stored as matrix of size ``(1000,300)``. The parameters ``half_window_size``
and ``negative_examples_cnt`` must have the same values as given to
``generate_cbow_data`` script.

The default values for parameters are:

* ``embedding_size`` - ``300``

* ``half_window_size`` - ``3``

* ``negative_examples_cnt`` - ``30``

See `README.nn.rst </README.nn.rst>`_  for more details about the network.

The script produces the following file in ``mdsd`` directory:
``mdsd.cbow.embedding.weights.npy``. After the script finishes successfully,
the files  in ``mdsd`` directory are:

::
    +---mdsd
        |  mdsd.csv
        |  wordfreq.json
        |  wordfreq.csv
        |  word2index.pck
        |  index2word.pck
        |  mdsd.indexed.csv
        |  mdsd.csv.cbow.data.input.000001.npy
        |  mdsd.csv.cbow.data.input.000002.npy
        |  ..
        |  mdsd.csv.cbow.data.input.NNNNNN.npy
        |  mdsd.csv.cbow.data.output.000001.npy
        |  mdsd.csv.cbow.data.output.000002.npy
        |  ..
        |  mdsd.csv.cbow.data.output.NNNNNN.npy
        |  mdsd.cbow.embedding.weights.npy
        +---books
        ..


wordvec_nn
^^^^^^^^^^

.. code-block:: bash

    $ python3 -m wordvec_nn [path_to_mdsd_dir] [nearest_neighbors_cnt] [kdtree_leafsize]

The ``wordvec_nn`` script reads ``mdsd.cbow.embedding.weights.npy`` file from
``mdsd`` directory (created by ``train_cbow`` script), and finds nearest
"neighbors" for each word (in terms of metric distance between corresponding
embedding vectors). This is not a proper clustering. However, it allows to see
the relations between words at-a-glance.

Finding of nearest neighbors of each word is done with
`k-d tree <https://en.wikipedia.org/wiki/K-d_tree>`. Exactly
``nearest_neighbors_cnt`` neighboring words are found for every word. The parameter
``kdtree_leafsize`` may be used to optimize the searching process.

The default values for parameters are:

* ``nearest_neighbors_cnt`` - ``100``

* ``kdtree_leafsize`` - ``16``

The script produces the following files in ``mdsd`` directory:

* ``mdsd.cbow.wordvec.closest.neighbors.npy`` - matrix of size
``(num_of_words, nearest_neighbors_cnt)``, where for each word the indexes of
neighboring words are stored as column

* ``mdsd.cbow.wordvec.closest.neighbors.csv`` - CSV file with the following
columns: ``Word``, ``Nearest Word 1``, ..., ``Nearest Word K``, where K is equal
to ``nearest_neighbors_cnt``; the words are stored in plain text.

After the script finishes successfully, the files  in ``mdsd`` directory are:

::
    +---mdsd
        |  mdsd.csv
        |  wordfreq.json
        |  wordfreq.csv
        |  word2index.pck
        |  index2word.pck
        |  mdsd.indexed.csv
        |  mdsd.csv.cbow.data.input.000001.npy
        |  mdsd.csv.cbow.data.input.000002.npy
        |  ..
        |  mdsd.csv.cbow.data.input.NNNNNN.npy
        |  mdsd.csv.cbow.data.output.000001.npy
        |  mdsd.csv.cbow.data.output.000002.npy
        |  ..
        |  mdsd.csv.cbow.data.output.NNNNNN.npy
        |  mdsd.cbow.embedding.weights.npy
        |  mdsd.cbow.wordvec.closest.neighbors.npy
        |  mdsd.cbow.wordvec.closest.neighbors.csv
        +---books
        ..


References
----------

Blitzer J., Dredze M., Pereira F. "Biographies, Bollywood, Boom-boxes and 
Blenders: Domain Adaptation for Sentiment Classification.", Association of
Computational Linguistics (ACL), 2007

Mikolov T., Chen K., Corrado G., Dean J. "Efficient Estimation of Word
Representations in Vector Space", https://arxiv.org/abs/1301.3781

Mikolov T., Sutskever I., Chen K., Corrado G., Dean J. "Distributed
Representations of Words and Phrases and their Compositionality",
https://arxiv.org/abs/1310.4546

Mnih A., Kavukcuoglu K. "Learning word embeddings efficiently with
noise-contrastive estimation", Advances in Neural Information Processing
Systems 26, 2265-2273, 2013
