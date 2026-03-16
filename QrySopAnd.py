"""
The AND operator for all retrieval models.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

from QrySop import QrySop
from RetrievalModelBM25 import RetrievalModelBM25
from RetrievalModelRankedBoolean import RetrievalModelRankedBoolean
from RetrievalModelUnrankedBoolean import RetrievalModelUnrankedBoolean


class QrySopAnd(QrySop):
    """
    The AND operator for all retrieval models.
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self):
        QrySop.__init__(self)		# Inherit from QrySop

    def docIteratorHasMatch(self, r):
        """
        Indicates whether the query has a match.
        For AND: true iff all query arguments match the same document.

        r: The retrieval model that determines what is a match.
        Returns True if the query matches, otherwise False.
        """
        return self.docIteratorHasMatchAll(r)

    def getScore(self, retrievalModel):
        """
        Get a score for the document that docIteratorHasMatch matched.

        retrievalModel: retrieval model parameters

        Returns the document score.

        throws IOException: Error accessing the Lucene index
        """

        if isinstance(retrievalModel, RetrievalModelUnrankedBoolean):
            return self.__getScoreBoolean(retrievalModel)
        elif isinstance(retrievalModel, RetrievalModelRankedBoolean):
            return self.__getScoreRankedBoolean(retrievalModel)
        elif isinstance(retrievalModel, RetrievalModelBM25):
            return self.__getScoreRankedBoolean(retrievalModel)
        else:
            raise Exception('{}.{} does not support {}'.format(
                self.__class__.__name__,
                sys._getframe().f_code.co_name,
                retrievalModel.__class__.__name__))

    def __getScoreBoolean(self, r):
        """
        getScore for Boolean retrieval models.
        For AND: 1.0 if all arguments match the document, else 0.0.

        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index
        """
        docid = self.docIteratorGetMatch()

        for q_i in self._args:
            if not (q_i.docIteratorHasMatch(r) and
                    q_i.docIteratorGetMatch() == docid):
                return 0.0

        return 1.0

    def __getScoreRankedBoolean(self, r):
        """
        getScore for Ranked Boolean AND: minimum of argument scores (score is
        limited by the rarest term). Matches common reference implementations.
        """
        docid = self.docIteratorGetMatch()
        min_score = None
        for q_i in self._args:
            if (q_i.docIteratorHasMatch(r) and
                    q_i.docIteratorGetMatch() == docid):
                s = q_i.getScore(r)
                min_score = s if min_score is None else min(min_score, s)
        return min_score if min_score is not None else 0.0
