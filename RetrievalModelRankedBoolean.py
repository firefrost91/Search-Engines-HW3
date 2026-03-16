"""
Define and store data for the Ranked Boolean Retrieval Model.
Documents are scored by term frequency (tf) and combined via sum (OR) or sum (AND).
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

from RetrievalModel import RetrievalModel


class RetrievalModelRankedBoolean(RetrievalModel):
    """
    Define and store data for the Ranked Boolean Retrieval Model.
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        RetrievalModel.__init__(self)		# Inherit from RetrievalModel
        self.defaultQrySop = '#AND'
