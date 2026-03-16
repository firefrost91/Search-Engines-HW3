"""
Get an initial ranking for a set of queries. The ranking may come
from an .inRank file or from a bag-of-words ranker (ranked and
unranked boolean, Indri, BM25).
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import itertools

from collections import OrderedDict

import Util

from Idx import Idx
from QryParser import QryParser
from Ranking import Ranking
from RetrievalModelBM25 import RetrievalModelBM25
from RetrievalModelRankedBoolean import RetrievalModelRankedBoolean
from RetrievalModelUnrankedBoolean import RetrievalModelUnrankedBoolean

class Ranker:
    """
    Get an initial ranking for a set of queries. The ranking may
    come from an .inRank file or from a bag-of-words ranker (ranked
    and unranked boolean, Indri, BM25).
    """


    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        self._model = None
        self._inRank_path = None
        self._max_results = 1000       		# default

        if 'outputLength' in parameters:
            self._max_results = parameters['outputLength']

        if 'type' not in parameters:
            raise AttributeError('Missing parameter: type')

        if parameters['type'] == 'inRankFile':
            self._inRank_path = parameters['inRankFile:Path']
        elif parameters['type'] == 'BM25':
            self._model = RetrievalModelBM25(parameters)
        elif parameters['type'] == 'RankedBoolean':
            self._model = RetrievalModelRankedBoolean(parameters)
        elif parameters['type'] == 'UnrankedBoolean':
            self._model = RetrievalModelUnrankedBoolean(parameters)
        else:
            raise AttributeError('Unknown type: {parameters["type"]}')


    def execute(self, batch):
        """
        Get rankings for each query. Any prior ranking is ignored.

        batch: A dict of {qid: {'qstring': qstring } ... }

        Return a dict of {qid: {'qstring': qstring,
                                'ranking': [(score, externalId)] ...}
                          ... }
        """
        if self._model is not None:
            return(self.get_rankings_bow(batch))
        elif self._inRank_path is not None:
            for qid, ranking in Util.read_rankings(self._inRank_path).items():
                batch[qid]['ranking'] = ranking
            return(batch)
        else:
            raise Exception('Error: Ranker does not know how to rank')
                        

    def get_rankings_bow(self, batch):
        """
        Add  rankings for each query to the batch object. Each ranking is
        a list of (score, externalId) tuples.
        
        batch: A dict of {query_id: {'qstring': query_string}}.
        """
        for qid in batch:
            # Prepare to evaluate a query
            qstring = batch[qid]["qstring"]
            print(f'{qid}: {qstring}', flush=True)
            qstring = f'{self._model.defaultQrySop}({qstring})'
            q = QryParser.getQuery(qstring)
            print(f'    ==> {str(q)}', flush=True)
            q.initialize(self._model)
            ranking = Ranking(self._max_results)

            # Evaluate the query. Each pass of the loop finds
            # one matching document. Skip if we already added this docid (avoid duplicates).
            seen_docids = set()
            while(q.docIteratorHasMatch(self._model)):
                docid = q.docIteratorGetMatch()
                if docid in seen_docids:
                    q.docIteratorAdvancePast(docid)
                    continue
                seen_docids.add(docid)
                score = q.getScore(self._model)
                q.docIteratorAdvancePast(docid)
                ranking.add(docid, score)

            rlist = ranking.get_ranking()
            batch[qid]['ranking'] = rlist
            # #region agent log
            if int(qid) == 169 and len(rlist) >= 7:
                try:
                    _lp = "/Users/ansh/Desktop/Search Engines/.cursor/debug.log"
                    import json, time
                    boundary = [(r[0], r[1]) for r in rlist[-7:]]
                    with open(_lp, "a") as _f:
                        _f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"Ranker.py:get_rankings_bow","message":"ranking boundary q169","data":{"qid":qid,"last7":boundary},"timestamp":int(time.time()*1000)}) + "\n")
                except Exception: pass
            # #endregion

        return(batch)
