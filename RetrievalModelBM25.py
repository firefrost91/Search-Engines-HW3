"""
Define and store data for the BM25 Retrieval Model.
Parameters: k1 (term frequency saturation), b (document length normalization).
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from RetrievalModel import RetrievalModel


class RetrievalModelBM25(RetrievalModel):
    """
    Define and store data for the BM25 Retrieval Model.
    Parameters k1 and b are read from the ranker config (e.g. task_1:ranker).
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        RetrievalModel.__init__(self)		# Inherit from RetrievalModel
        self.defaultQrySop = '#SUM'
        # Read from param file: BM25:k_1 and BM25:b (fallback to k1, b)
        self._k1 = float(parameters.get('BM25:k_1', parameters.get('k1', 1.2)))
        self._b = float(parameters.get('BM25:b', parameters.get('b', 0.75)))
