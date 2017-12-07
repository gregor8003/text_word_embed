Training data
-------------


Each text document is scanned with sliding window of ``2 * half_window_size + 1``.
For each word at the center of window, the following training data is
generated:

* positive examples, that consist of pairs ``(center_word, neigh_word)``, where ``neigh_word`` comes from sliding window; corresponding output is generated as well

* negative examples, that consist of pairs ``(center_word, another_word)``, where ``another_word`` comes from outside of sliding window (but still from the same text document); corresponding output is generated as well

For each positive example, ``negative_examples_cnt`` negatives examples are generated.
Therefore, in total, ``len(positive_examples) * negative_examples_cnt`` negative
examples are generated.

Matching output data is generated as follows: for positive training example,
it is a pair of integers ``(1,1)``, and for negative training example, it is
a pair of integers ``(1,0)``.

For instance, sliding window ``[3 2 7 10 5 6 4]`` is of length ``7`` and has
``half_window_size`` of 3. The following training data are generated:

* positive examples: ``(10,3),(10,2),(10,7),(10,5),(10,6),(10,4)``

* negative examples: ``(10,w1),...,(10,wK)``, where wK is a word from outside sliding window; K is equal to ``len(positive_examples)*negative_examples_cnt``
