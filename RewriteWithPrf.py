"""
Rewrite queries using pseudo relevance feedback (PRF).
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import math

import Util

from Idx import Idx
from QryParser import QryParser


class RewriteWithPrf:
    """
    Rewrite queries using pseudo relevance feedback.
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        if 'prf:algorithm' not in parameters:
            raise Exception('Error: Missing parameter prf:algorithm.')

        if 'prf:numDocs' not in parameters:
            raise Exception('Error: Missing parameter prf:numDocs.')

        if 'prf:numTerms' not in parameters:
            raise Exception('Error: Missing parameter prf:numTerms.')

        self._algorithm = str(parameters['prf:algorithm']).lower()
        if self._algorithm not in ('okapi', 'rm3'):
            raise Exception(f'Error: Unknown prf:algorithm {parameters["prf:algorithm"]}')

        self._num_docs = int(parameters['prf:numDocs'])
        self._num_terms = int(parameters['prf:numTerms'])
        # Support the homework alias prf:expansionField for the common case
        # where expansion reads from and writes to the same field.
        default_field = parameters.get('prf:expansionField', 'body')
        self._field_in = parameters.get('prf:expansionFieldIn', default_field)
        self._field_out = parameters.get('prf:expansionFieldOut', default_field)
        self._qry_out_path = parameters.get('prf:expansionQueryFile')
        self._orig_weight = parameters.get('prf:rm3:origWeight')
        if self._orig_weight is not None:
            self._orig_weight = float(self._orig_weight)

        # Cache collection-wide values used by PRF.
        self._num_docs_total = float(Idx.getNumDocs())
        self._doc_count_field = float(Idx.getDocCount(self._field_in))
        self._sum_field_len = float(Idx.getSumOfFieldLengths(self._field_in))
        self._avg_doc_len = (self._sum_field_len / self._doc_count_field
                             if self._doc_count_field > 0 else 1.0)
        if self._avg_doc_len <= 0:
            self._avg_doc_len = 1.0

        self._rsj_cache = {}


    def execute(self, batch):
        """
        Rewrite queries for each query in the batch using PRF.

        batch: A dict of {qid: {'qstring': qstring,
                                'ranking': [(score, externalId)] ...}
                          ... }
        """
        qry_out_lines = []

        for qid in batch:
            if 'ranking' not in batch[qid]:
                raise Exception('Error: Missing ranking for PRF rewriter.')

            ranking = batch[qid]['ranking']
            query_terms = self._tokenize_feedback_query(batch[qid]['qstring'])
            term_scores = self._score_terms(ranking, query_terms)
            top_terms = self._select_terms(term_scores)

            learned_query = self._build_learned_query(top_terms)
            if learned_query is None:
                learned_query = batch[qid]['qstring']
                expanded_query = batch[qid]['qstring']
            else:
                expanded_query = self._build_expanded_query(batch[qid]['qstring'], learned_query)

            if self._qry_out_path is not None:
                qry_out_lines.append(f'{qid}: {learned_query}')

            batch[qid]['qstring'] = expanded_query

        if self._qry_out_path is not None:
            Util.file_write_strings(self._qry_out_path, qry_out_lines)

        return batch


    def _build_expanded_query(self, original_qstring, learned_query):
        if self._algorithm != 'rm3':
            return learned_query

        if self._orig_weight is None or self._orig_weight <= 0:
            return learned_query

        if self._orig_weight >= 1:
            return original_qstring

        orig_subquery = self._wrap_query(original_qstring)
        exp_weight = 1.0 - float(self._orig_weight)
        return f'#WSUM( {self._orig_weight} {orig_subquery} {exp_weight} {learned_query} )'


    def _build_learned_query(self, term_scores):
        if not term_scores:
            return None

        if self._algorithm == 'rm3':
            total = sum(score for _, score in term_scores)
            if total <= 0:
                total = 1.0
            parts = []
            for term, score in term_scores:
                parts.append(f'{score / total} {self._format_term(term)}')
            return '#WSUM( ' + ' '.join(parts) + ' )'

        terms = [self._format_term(term) for term, _ in term_scores]
        return '#SUM( ' + ' '.join(terms) + ' )'


    def _format_term(self, term):
        if self._field_out and self._field_out != 'body':
            return f'{term}.{self._field_out}'
        return term


    def _is_valid_term(self, term):
        if term is None:
            return False
        if '.' in term or ',' in term:
            return False
        return term.isascii()


    def _okapi_term_score(self, tf, doc_len, term):
        rsj = self._rsj_cache.get(term)
        if rsj is None:
            df = float(Idx.getDocFreq(self._field_in, term))
            # PRF scores should use statistics from the expansion field rather
            # than the whole collection so title/url/inlink expansions are
            # weighted against the right document universe.
            doc_count = self._doc_count_field if self._doc_count_field > 0 else self._num_docs_total
            rsj = math.log((doc_count - df + 0.5) / (df + 0.5))
            self._rsj_cache[term] = rsj
        tf_weight = tf / (tf + 0.5 + (1.5 * (doc_len / self._avg_doc_len)))
        return tf_weight * rsj


    def _query_likelihood_log_weight(self, tv, doc_len, query_terms):
        """
        Return log p(q|d) with no smoothing. If any query term is absent,
        the document provides no evidence for RM3.
        """
        if not query_terms:
            return None

        log_weight = 0.0
        for term in query_terms:
            term_idx = tv.indexOfStem(term)
            if term_idx < 0:
                return None

            tf = float(tv.stemFreq(term_idx))
            if tf <= 0:
                return None

            log_weight += math.log(tf / doc_len)

        return log_weight


    def _score_terms(self, ranking, query_terms):
        term_scores = {}
        top_docs = ranking[:self._num_docs]
        if not top_docs:
            return term_scores

        doc_infos = []
        max_log_weight = None

        for _, external_id in top_docs:
            docid = Idx.getInternalDocid(external_id)
            tv = Idx.getTermVector(docid, self._field_in)
            if tv is None or tv.stemsLength() == 0:
                continue

            doc_len = tv.positionsLength()
            if doc_len <= 0:
                continue

            log_weight = None
            if self._algorithm == 'rm3':
                log_weight = self._query_likelihood_log_weight(tv, doc_len, query_terms)
                if log_weight is None:
                    continue
                if max_log_weight is None or log_weight > max_log_weight:
                    max_log_weight = log_weight

            doc_infos.append((tv, doc_len, log_weight))

        for tv, doc_len, log_weight in doc_infos:
            weight_doc = 1.0
            if self._algorithm == 'rm3':
                weight_doc = math.exp(log_weight - max_log_weight)

            for stem_i in range(1, tv.stemsLength()):
                term_obj = tv.stemString(stem_i)
                if term_obj is None:
                    continue
                term = str(term_obj)
                if not self._is_valid_term(term):
                    continue

                tf = float(tv.stemFreq(stem_i))
                if tf <= 0:
                    continue

                if self._algorithm == 'okapi':
                    score_term = self._okapi_term_score(tf, doc_len, term)
                else:
                    score_term = (tf / doc_len) * weight_doc

                term_scores[term] = term_scores.get(term, 0.0) + score_term

        return term_scores


    def _select_terms(self, term_scores):
        if not term_scores:
            return []

        ranked = sorted(term_scores.items(), key=lambda x: (-x[1], x[0]))
        print("\n--- Top Terms (descending by weight) ---")
        for term, score in ranked[:15]:
            print(f"{term}: {score}")
        ranked = ranked[:self._num_terms]
        ranked.sort(key=lambda x: (x[1], x[0]))
        return ranked


    def _wrap_query(self, qstring):
        qstring = qstring.strip()
        return f'#SUM( {qstring} )'


    def _tokenize_feedback_query(self, qstring):
        """
        Convert the current query string into stopword-removed, stemmed BOW
        terms for feedback scoring.
        """
        return QryParser.tokenizeString(QryParser.bowQuery(qstring))
