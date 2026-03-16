"""
The WINDOW/n operator for all retrieval models.
#WINDOW/n (a b c): all arguments occur in the document in any order
within a window of n consecutive terms.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from InvList import InvList
from QryIop import QryIop


class QryIopWindow(QryIop):
    """The WINDOW/n operator for all retrieval models."""

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, n):
        """
        Create a WINDOW/n query node.
        n: Window size (n consecutive positions must contain all terms).
        """
        QryIop.__init__(self)
        if n <= 0:
            raise ValueError("Window size n must be positive")
        self._n = n

    def evaluate(self):
        """
        Evaluate the query operator; the result is an internal inverted
        list. For each document, find all windows of n terms that contain
        all query terms (in any order).         Posting stores (docid, [window_start
        of each match]); tf = number of matches.
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

            # Collect position lists for this doc from all args
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
                match_starts = self._find_window_matches(position_lists, self._n)
                if match_starts:
                    self.invertedList.appendPosting(minDocid, match_starts)

            # Advance all args that are at minDocid
            for q_i in self._args:
                if (q_i.docIteratorHasMatch(None) and
                        q_i.docIteratorGetMatch() == minDocid):
                    q_i.docIteratorAdvancePast(minDocid)

        return

    def _find_window_matches(self, position_lists, n):
        """
        Find all windows where all terms appear within n consecutive positions.
        Returns the MAX position of each valid window (as per reference implementation).
        
        Algorithm (from slides):
        - If (max - min) < n: MATCH -> record max position, advance ALL pointers
        - If (max - min) >= n: NO MATCH -> advance only MIN pointer(s)
        """
        k = len(position_lists)
        if k == 0:
            return []
        if k == 1:
            return list(position_lists[0])

        idx = [0] * k
        matches = []

        while True:
            # Check if any list is exhausted
            if any(idx[i] >= len(position_lists[i]) for i in range(k)):
                break

            # Get current positions
            curr = [position_lists[i][idx[i]] for i in range(k)]
            minp = min(curr)
            maxp = max(curr)

            # Check if valid window: (max - min) < n
            if (maxp - minp) < n:
                # MATCH: Record MAX position and advance ALL pointers
                matches.append(maxp)
                for i in range(k):
                    idx[i] += 1
            else:
                # NO MATCH: Advance only the MIN pointer(s)
                min_indices = [i for i in range(k) if curr[i] == minp]
                for i in min_indices:
                    idx[i] += 1

        return matches