"""
The NEAR/n operator for all retrieval models.
#NEAR/n (a b c): all arguments occur in the document, in order,
with no more than n-1 terms between two adjacent terms.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from InvList import InvList
from QryIop import QryIop


class QryIopNear(QryIop):
    """The NEAR/n operator for all retrieval models."""

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, n):
        """
        Create a NEAR/n query node.
        n: Maximum gap between adjacent terms (n-1 terms may separate them).
        """
        QryIop.__init__(self)
        self._n = n

    def evaluate(self):
        """
        Evaluate the query operator; the result is an internal inverted
        list. For each document containing all terms in order with gaps <= n,
        the posting stores (docid, [start_pos of each match]); tf = num matches.
        """
        if len(self._args) > 0:
            self._field = getattr(self._args[0], '_field', None)
        self.invertedList = InvList(self._field)

        if len(self._args) == 0:
            return

        while True:
            minDocid = None
            for q_i in self._args:
                if q_i.docIteratorHasMatch(None):
                    q_iDocid = q_i.docIteratorGetMatch()
                    if minDocid is None or minDocid > q_iDocid:
                        minDocid = q_iDocid

            if minDocid is None:
                break

            position_lists = []
            all_have_doc = True
            for q_i in self._args:
                if (q_i.docIteratorHasMatch(None) and
                        q_i.docIteratorGetMatch() == minDocid):
                    position_lists.append(
                        list(q_i.docIteratorGetMatchPosting().positions))
                else:
                    all_have_doc = False
                    break

            if all_have_doc and len(position_lists) == len(self._args):
                match_starts = self._find_near_matches(position_lists, self._n)
                if match_starts:
                    self.invertedList.appendPosting(minDocid, match_starts)

            for q_i in self._args:
                if (q_i.docIteratorHasMatch(None) and
                        q_i.docIteratorGetMatch() == minDocid):
                    q_i.docIteratorAdvancePast(minDocid)

        return

    def _find_near_matches(self, position_lists, n):
        """
        Find all (p1, p2, ..., pk) with terms in order and p_{i+1} - p_i <= n.
        Returns list of last position of each match (one per match).
        """
        k = len(position_lists)
        if k == 0:
            return []
        if k == 1:
            return list(position_lists[0])

        match_positions = []

        def backtrack(idx, prev_pos):
            if idx == k:
                match_positions.append(prev_pos)
                return
            for p in position_lists[idx]:
                if idx > 0:
                    if p <= prev_pos:
                        continue
                    if p - prev_pos > n:
                        break
                backtrack(idx + 1, p)

        backtrack(0, -1)
        return match_positions
