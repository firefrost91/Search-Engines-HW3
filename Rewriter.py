"""
Rewrite queries using pseudo relevance feedback and other methods.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from RewriteWithPrf import RewriteWithPrf


class Rewriter:
    """
    Rewrite queries for a set of tasks (currently PRF).
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        if 'type' not in parameters:
            raise Exception('Error: Missing parameter type.')

        task_type = str(parameters['type']).lower()
        if task_type == 'prf':
            self._model = RewriteWithPrf(parameters)
        else:
            raise Exception(f'Error: Unknown type: {parameters["type"]}')


    def execute(self, batch):
        """
        Rewrite queries for each query in the batch.

        batch: A dict of {qid: {'qstring': qstring,
                                'ranking': [(score, externalId)] ...}
                          ... }
        """
        return self._model.execute(batch)
