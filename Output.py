"""
Write results in various formats.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from TeIn import TeIn

class Output:

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        self._type = parameters['type']
        self._outputPath = parameters['outputPath']
        self._outputLength = parameters['outputLength']


    def close(self):
         if '_teIn' in vars(self):
             self._teIn.close()


    def execute(self, batch):
        """
        Write the output about the batch to a file

        batch: A dict of {qid: {'qstring': qstring,
                                'ranking': [(score, externalId) ...]}
                          ... }
        """
        if self._type == 'trec_eval':
            teIn = TeIn(self._outputPath, self._outputLength)
            for qid in batch:
                # Ranking is already (score desc, internal docid asc) from Ranker; preserve order.
                ranking = batch[qid].get('ranking') or []
                teIn.appendQuery(qid, ranking, 'reference')
            teIn.close()
#
# You will need to add another format for HW5.
#
        else:
            raise Exception('Error: Unknown Output format')
            
        return(batch)

