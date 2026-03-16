"""
The OR operator for all retrieval models.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import sys

from QrySop import QrySop
from RetrievalModelBM25 import RetrievalModelBM25
from RetrievalModelRankedBoolean import RetrievalModelRankedBoolean
from RetrievalModelUnrankedBoolean import RetrievalModelUnrankedBoolean


class QrySopOr(QrySop):
    """
    The OR operator for all retrieval models.
    """

    # -------------- Methods (alphabetical) ---------------- #


    def __init__(self):
        QrySop.__init__(self)		# Inherit from QrySop


    def docIteratorHasMatch(self, r):
        """
        Indicates whether the query has a match.

        r: The retrieval model that determines what is a match.
        Returns True if the query matches, otherwise False.
        """
        return self.docIteratorHasMatchMin(r)


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

        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        throws IOException: Error accessing the Lucene index
        """

        # Return the maximum of its query argument scores.  The
        # document matches, but we don't know which query arguments
        # match, so check each query argument.
        score = 0.0;
        docid = self.docIteratorGetMatch()

        for q_i in self._args:
            if (q_i.docIteratorHasMatch(r) and
                q_i.docIteratorGetMatch() == docid):
                score = max(score, q_i.getScore(r))

        return(score)

    def __getScoreRankedBoolean(self, r):
        """
        getScore for Ranked Boolean: MAX of argument scores for the current document.
        """
        score = 0.0
        docid = self.docIteratorGetMatch()
        for q_i in self._args:
            if (q_i.docIteratorHasMatch(r) and
                    q_i.docIteratorGetMatch() == docid):
                score = max(score, q_i.getScore(r))
        return score
