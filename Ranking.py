"""
Create, access, and manipulate document score lists.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.


from Idx import Idx

class Ranking:
    """
    A Ranking object stores a search ranking and returns the top n.
    """

    class Entry:
        """
        A utility class to create an <externalDocid, score> object
        for the _ranking list. internalId is kept for tie-breaking.
        """

        def __init__(self, score, externalId, internalId):
            self.score = score
            self.externalId = externalId
            self.internalId = internalId

        def __lt__(self, other):
            return (self.score < other.score or
                    (self.score == other.score and self.internalId < other.internalId))

        def __str__(self):
            return(f'{self.externalId} {self.score}')


    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, n):
        """Create an empty list that can store a ranking."""
        self._max_size = n
        self._ranking = []


    def __len__(self):
        return(len(self._ranking))


    def add(self, internalId, score):
        """
        Add a document to the ranking (unsorted).
        internalId: An internal document id (Lucene docid).
        score: The document score (stored raw; do not round before sorting).
        """
        externalId = Idx.getExternalDocid(internalId)
        self._ranking.append(self.Entry(float(score), externalId, internalId))
        

    def get_ranking(self):
        """
        Get a ranked list: sort by score (desc), then internal docid (asc); dedupe; then truncate.
        Max results cap is applied after sorting so per-query output is consistent.
        Returns list of (score, externalId) for trec_eval.
        """
        results_qid = [(r.score, r.externalId, r.internalId) for r in self._ranking]
        # Sort: score desc, then tie-break by external docid alphabetically (design guide).
        results_qid.sort(key=lambda r: (-r[0], r[1]))
        results_qid = [(r[0], r[1]) for r in results_qid]
        # Deduplicate by external docid (keep first = highest score per doc)
        seen = set()
        deduped = []
        for r in results_qid:
            if r[1] not in seen:
                seen.add(r[1])
                deduped.append(r)
        # Truncate after sort and dedup (consistent per-query cap)
        return deduped[0:self._max_size]


