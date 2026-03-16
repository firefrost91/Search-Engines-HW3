"""
The SUM operator for all retrieval models.
Return a document if at least one query argument occurs; score = sum of argument scores.
Used by BM25.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

from QrySop import QrySop


class QrySopSum(QrySop):
    """
    The SUM operator for all retrieval models.
    Return a document if at least one argument matches; combine scores by summation.
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self):
        QrySop.__init__(self)		# Inherit from QrySop

    def docIteratorHasMatch(self, r):
        """
        True iff at least one query argument matches the document.
        """
        return self.docIteratorHasMatchMin(r)

    def getScore(self, retrievalModel):
        """
        Score = sum of scores from all arguments that match the current document.
        """
        score = 0.0
        docid = self.docIteratorGetMatch()
        for q_i in self._args:
            if (q_i.docIteratorHasMatch(retrievalModel) and
                    q_i.docIteratorGetMatch() == docid):
                score += q_i.getScore(retrievalModel)
        return score
