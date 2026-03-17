"""
The SCORE operator for all retrieval models.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import math
import sys

from Idx import Idx
from QrySop import QrySop
from RetrievalModelBM25 import RetrievalModelBM25
from RetrievalModelRankedBoolean import RetrievalModelRankedBoolean
from RetrievalModelUnrankedBoolean import RetrievalModelUnrankedBoolean


class QrySopScore(QrySop):
    """
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
        return(self.docIteratorHasMatchFirst(r))


    def getScore(self, r):
        """
        Get a score for the document that docIteratorHasMatch matched.
        UnrankedBoolean: 1.0 for matched docs (constant; not tf, not child scores).
        RankedBoolean: tf (match count) from Iop. BM25: IDF * tf formula.
        """
        if isinstance(r, RetrievalModelUnrankedBoolean):
            return self.__getScoreUnrankedBoolean(r)
        elif isinstance(r, RetrievalModelRankedBoolean):
            return self.__getScoreRankedBoolean(r)
        elif isinstance(r, RetrievalModelBM25):
            return self.__getScoreBM25(r)
        else:
            raise Exception(
                '{} does not support the #SCORE operator.'.format(
                    r.__class__.__name__))


    def __getScoreUnrankedBoolean(self, r):
        """
        getScore for the Unranked retrieval model.
        If the doc matches: return exactly 1.0 (constant; never tf or child scores).
        If no match: return 0.0.
        """
        if not self.docIteratorHasMatchCache():
            return 0.0
        return 1.0

    def __getScoreRankedBoolean(self, r):
        """
        getScore for the Ranked Boolean retrieval model.
        Score = match count (tf) of the current document from the Iop argument.
        #SCORE(#NEAR/n(...)) or #SCORE(#WINDOW/n(...)) turns proximity match count into the score.

        r: The retrieval model that determines how scores are calculated.
        Returns the document score.
        """
        q_iop = self._args[0]
        if not q_iop.docIteratorHasMatch(r):
            return 0.0
        # Ensure we score the doc the iterator is on (same docid as cached at this SCORE node)
        docid = self.docIteratorGetMatch()
        if q_iop.docIteratorGetMatch() != docid:
            return 0.0
        posting = q_iop.docIteratorGetMatchPosting()
        return float(posting.tf)

    def __getScoreBM25(self, r):
        """
        getScore for the BM25 retrieval model.
        score = IDF * (tf * (k1+1)) / (tf + k1 * (1 - b + b * doclen/avgdoclen))
        IDF = max(0, log((N - df + 0.5) / (df + 0.5)))
        where N is the number of documents that contain this field.
        """
        q_iop = self._args[0]
        if not q_iop.docIteratorHasMatch(r):
            return 0.0
        docid = q_iop.docIteratorGetMatch()
        posting = q_iop.docIteratorGetMatchPosting()
        tf = float(posting.tf)
        field = q_iop._field
        N = float(Idx.getDocCount(field))
        df = float(q_iop.getDf())
        doclen = float(Idx.getFieldLength(field, docid))
        sum_len = float(Idx.getSumOfFieldLengths(field))
        doc_count_field = float(Idx.getDocCount(field))
        # Average doc length over docs that have this field (matches InspectIndex / common BM25)
        avgdoclen = sum_len / doc_count_field if doc_count_field > 0 else 1.0
        if avgdoclen <= 0:
            avgdoclen = 1.0
        k1 = r._k1
        b = r._b
        rsj = (N - df + 0.5) / (df + 0.5)
        idf = max(0.0, math.log(rsj)) if rsj > 0.0 else 0.0
        norm = 1.0 - b + b * (doclen / avgdoclen)
        return idf * (tf * (k1 + 1.0)) / (tf + k1 * norm)

    def initialize(self, r):
        """
        Initialize the query operator (and its arguments), including any
        internal iterators.  If the query operator is of type QryIop, it
        is fully evaluated, and the results are stored in an internal
        inverted list that may be accessed via the internal iterator.

        r: A retrieval model that guides initialization.
        throws IOException: Error accessing the Lucene index.
        """
        q = self._args[ 0 ]
        q.initialize(r)
