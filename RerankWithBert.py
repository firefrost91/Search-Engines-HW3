"""
BERT-based reranker for a passage-level reranking pipeline.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from Idx import Idx


class RerankWithBERT:
    """
    Rerank documents using a BERT-based passage-level scoring model.

    For each document in the initial ranking, the body text is split
    into overlapping passages. Each passage is scored as a (query,
    passage) pair by BERT. Passage scores are aggregated into a single
    document score using firstp, maxp, or avgp.
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        """
        Initialize the BERT reranker.

        Required parameters:
            bertrr:modelPath        Path to HuggingFace model directory.
            bertrr:psgLen           Max body tokens per passage (int).
            bertrr:psgCnt           Max passages per document (int).
            bertrr:scoreAggregation Aggregation method: firstp, avgp, or maxp.

        Optional parameters:
            bertrr:psgStride        Stride between passage starts (int, default psgLen).
            bertrr:maxTitleLength   Max title tokens to prepend (int, default 0).
        """
        required = [
            'bertrr:modelPath',
            'bertrr:psgLen',
            'bertrr:psgCnt',
            'bertrr:scoreAggregation',
        ]
        for key in required:
            if key not in parameters:
                raise Exception(f'Error: Missing parameter {key}.')

        self._psg_len           = int(parameters['bertrr:psgLen'])
        # psgStride is optional; default to psgLen (non-overlapping) when absent.
        self._psg_stride        = int(parameters.get('bertrr:psgStride', self._psg_len))
        self._psg_cnt           = int(parameters['bertrr:psgCnt'])
        self._max_title_length  = int(parameters.get('bertrr:maxTitleLength', 0))
        self._score_aggregation = str(parameters['bertrr:scoreAggregation']).lower()

        if self._score_aggregation not in ('firstp', 'avgp', 'maxp'):
            raise Exception(
                f'Error: Unknown bertrr:scoreAggregation '
                f'"{parameters["bertrr:scoreAggregation"]}". '
                f'Expected firstp, avgp, or maxp.')

        model_path = str(parameters['bertrr:modelPath'])
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._bert_model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=1)
        self._bert_model.eval()


    def rerank(self, batch):
        """
        Score and rerank documents for each query.

        batch: {qid: {'qstring': str, 'ranking': [(score, ext_id), ...]}, ...}
        Returns the same structure with updated 'ranking' lists sorted by
        descending BERT score, then ascending external doc id (tie-break).
        """
        results = {qid: dict(batch[qid]) for qid in batch}

        for qid in batch:
            q_str = batch[qid]['qstring']
            scored_docs = []

            for _score, ext_id in batch[qid]['ranking']:
                try:
                    int_id = Idx.getInternalDocid(ext_id)
                except Exception:
                    continue

                body_str  = Idx.getAttribute('body-string',  int_id) or ''
                title_str = Idx.getAttribute('title-string', int_id) or ''

                passages = self._get_passages(body_str, title_str)
                score    = self._score_document(q_str, passages)
                scored_docs.append((score, ext_id))

            # Sort: descending score, then ascending ext_id (HW1/HW2 convention).
            scored_docs.sort(key=lambda x: (-x[0], x[1]))
            results[qid]['ranking'] = scored_docs

        return results


    def _get_passages(self, body_str, title_str):
        """
        Split body text into overlapping passages; optionally prepend title.

        A new passage is only created if its content extends beyond the
        previous passage (i.e., the previous passage does not fully cover it).
        bertrr:psgLen applies only to body tokens; title prepend is additional.

        Returns a list of passage strings (possibly empty).
        """
        body_tokens = body_str.split()
        if not body_tokens:
            return []

        raw_passages = []
        i = 0
        while i < len(body_tokens) and len(raw_passages) < self._psg_cnt:
            raw_passages.append(body_tokens[i : i + self._psg_len])
            if i + self._psg_len >= len(body_tokens):
                break
            i += self._psg_stride

        if self._max_title_length > 0:
            title_tokens = title_str.split()[:self._max_title_length]
            return [' '.join(title_tokens + p) for p in raw_passages]
        else:
            return [' '.join(p) for p in raw_passages]


    def _score_document(self, q_str, passages):
        """
        Score a document from its passage strings using BERT.

        Returns the aggregated float score, or 0.0 if there are no passages.
        """
        if not passages:
            return 0.0

        passage_scores = []
        for psg_str in passages:
            tensors = self._tokenizer.encode_plus(
                [q_str, psg_str],
                add_special_tokens=True,
                max_length=512,
                truncation='only_second',
                return_tensors='pt',
            )
            with torch.no_grad():
                passage_scores.append(
                    self._bert_model(**tensors).logits.data.item())

        if self._score_aggregation == 'firstp':
            return passage_scores[0]
        elif self._score_aggregation == 'maxp':
            return max(passage_scores)
        else:  # avgp
            return sum(passage_scores) / len(passage_scores)
