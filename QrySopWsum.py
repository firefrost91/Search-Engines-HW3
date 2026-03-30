"""
The WSUM operator for all retrieval models.
Return a document if at least one query argument occurs; score = weighted sum of argument scores.
Used by BM25.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from QrySop import QrySop


class QrySopWsum(QrySop):
    """
    The WSUM operator for all retrieval models.
    Return a document if at least one argument matches; combine scores by weighted summation.
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self):
        QrySop.__init__(self)		# Inherit from QrySop
        self._weights = []		# weight per argument, same order as _args

    def appendArg(self, q, weight=1.0):
        """
        Append an argument with an optional weight. Used by parser and _optimizeSumToWsum.
        """
        self._weights.append(weight)
        from Qry import Qry
        Qry.appendArg(self, q)

    def docIteratorHasMatch(self, r):
        """
        True iff at least one query argument matches the document.
        """
        return self.docIteratorHasMatchMin(r)

    def getScore(self, retrievalModel):
        """
        Score = sum of (weight_i * score_i) for all arguments that match the current document.
        """
        score = 0.0
        docid = self.docIteratorGetMatch()
        for i, q_i in enumerate(self._args):
            if (q_i.docIteratorHasMatch(retrievalModel) and
                    q_i.docIteratorGetMatch() == docid):
                w = self._weights[i] if i < len(self._weights) else 1.0
                score += w * q_i.getScore(retrievalModel)
        return score
 
