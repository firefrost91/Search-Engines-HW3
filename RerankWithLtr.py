"""
Access and manage a feature-based learning-to-rank (Ltr) reranker.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import math
import subprocess

import PyLu                  # Used to access RankLib
import Util                  # Used to read and write files

from Idx import Idx
from QryParser import QryParser         # Parse queries

class RerankWithLtr:
    """
    Access and manage a feature-based learning-to-rank (Ltr) reranker.
    """

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):

        # Store the parameters for the LTR reranker.
        self.params = parameters

        # BM25 / QL parameters for feature computation.
        self.bm25_b = float(parameters.get('ltr:BM25:b', 0.75))
        self.bm25_k1 = float(parameters.get('ltr:BM25:k_1', 1.2))
        self.ql_mu = float(parameters.get('ltr:QL:mu', 2500))

        # LTR toolkit configuration.
        self.toolkit = parameters.get('ltr:toolkit', 'RankLib').lower()
        self.model_file = parameters.get('ltr:modelFile')

        # Feature configuration.
        disable_str = parameters.get('ltr:featureDisable')
        if disable_str is None or disable_str.strip() == '':
            self.disabled_features = set()
        else:
            self.disabled_features = set(
                int(f.strip()) for f in disable_str.split(',') if f.strip().isdigit()
            )

        # Train model once during initialization.
        self._train_model()
        

    def rerank(self, batch):
        """
        Update the results for a set of queries with new scores.

        batch: A dict of {qid: {'qstring': qstring,
                                'ranking': [(score, externalId) ...]}
                          ... }
        """

        # Generate testing feature vectors for all <q, d> pairs in batch.
        testing_path = self.params.get('ltr:testingFeatureVectorsFile')
        scores_path = self.params.get('ltr:testingDocumentScores')

        flat_pairs = []  # (qid, externalId) in the same order as feature vectors
        lines = []

        # Process queries in ascending numeric qid order for determinism.
        for qid in sorted(batch.keys(), key=lambda x: int(x)):
            qstring = batch[qid]['qstring']
            bow_q = QryParser.bowQuery(qstring)
            q_stems = QryParser.tokenizeString(bow_q)

            # Collect raw feature dicts for this query.
            per_query_feats = []
            docids = []
            eids = []

            for score, eid in batch[qid]['ranking']:
                internal = Idx.getInternalDocid(eid)
                if internal < 0:
                    continue
                feats = self._compute_standard_features(q_stems, internal)
                feats = self._apply_feature_disable(feats)
                per_query_feats.append(feats)
                docids.append(internal)
                eids.append(eid)
                flat_pairs.append((qid, eid))

            # Normalize if using SVMrank.
            if self.toolkit == 'svmrank':
                per_query_feats = self._normalize_per_query(per_query_feats)
            else:
                # For RankLib, fill missing features with 0.0, no normalization required.
                per_query_feats = self._fill_missing_with_zero(per_query_feats)

            # Convert to feature-vector file lines (label is 0 for testing).
            for feats, eid in zip(per_query_feats, eids):
                line = self._format_feature_vector_line(
                    label=0,
                    qid=qid,
                    feats=feats,
                    external_id=eid
                )
                lines.append(line)

        if testing_path is not None:
            Util.file_write_strings(testing_path, lines)

        # Call toolkit to score the testing feature vectors.
        self._score_test_vectors(testing_path, scores_path)

        # Read scores and update rankings.
        scores = Util.file_read_strings(scores_path)
        if scores is None:
            # If scoring failed, fall back to original rankings.
            return batch

        # Parse toolkit-specific score file.
        parsed_scores = []
        if self.toolkit == 'svmrank':
            # One score per line, same order as feature vectors.
            parsed_scores = [float(s) for s in scores if s.strip() != '']
        else:
            # RankLib: qid \t rank \t score per line, same order as feature vectors.
            for line in scores:
                if not line.strip():
                    continue
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                parsed_scores.append(float(parts[2]))

        # Assign scores back to (qid, eid) pairs and rebuild rankings.
        idx = 0
        new_batch = {}
        by_qid = {}

        for (qid, eid), s in zip(flat_pairs, parsed_scores):
            by_qid.setdefault(qid, [])
            by_qid[qid].append((s, eid))
            idx += 1

        for qid in batch:
            if qid in by_qid:
                # Sort documents for this query by new score descending.
                new_ranking = sorted(by_qid[qid], key=lambda x: x[0], reverse=True)
            else:
                # No scores generated (e.g., all docs filtered out); keep original.
                new_ranking = batch[qid]['ranking']

            new_batch[qid] = {
                'qstring': batch[qid]['qstring'],
                'ranking': new_ranking
            }

        return new_batch

    # -------------- Internal helpers ----------------------- #

    def _train_model(self):
        """
        Generate training feature vectors and call the configured
        toolkit to learn a model.
        """
        train_qry_path = self.params.get('ltr:trainingQueryFile')
        train_qrels_path = self.params.get('ltr:trainingQrelsFile')
        train_feat_path = self.params.get('ltr:trainingFeatureVectorsFile')

        if (train_qry_path is None or train_qrels_path is None or
                train_feat_path is None or self.model_file is None):
            # Nothing to train; leave model uninitialized.
            return

        queries = Util.read_queries(train_qry_path)
        qrels = Util.read_qrels(train_qrels_path)

        # Map qid -> list of (eid, label).
        per_qid = {}
        for qid, _, eid, rel in qrels:
            # Handle 2-point and 5-point scales; treat -2 (spam) as 0.
            label = float(rel.strip())
            if label == -2:
                label = 0.0
            per_qid.setdefault(qid.strip(), []).append((eid.strip(), label))

        lines = []

        # For each training query, generate all feature vectors.
        for qid in sorted(per_qid.keys(), key=lambda x: int(x)):
            raw_q = queries.get(qid, '')
            q_stems = QryParser.tokenizeString(raw_q)

            # Collect raw feature dicts for this query.
            per_query_feats = []
            labels = []
            eids = []

            for eid, label in per_qid[qid]:
                internal = Idx.getInternalDocid(eid)
                if internal < 0:
                    continue
                feats = self._compute_standard_features(q_stems, internal)
                feats = self._apply_feature_disable(feats)
                per_query_feats.append(feats)
                labels.append(label)
                eids.append(eid)

            if not per_query_feats:
                continue

            # Normalize if SVMrank, otherwise just fill missing with 0.0.
            if self.toolkit == 'svmrank':
                per_query_feats = self._normalize_per_query(per_query_feats)
            else:
                per_query_feats = self._fill_missing_with_zero(per_query_feats)

            for feats, label, eid in zip(per_query_feats, labels, eids):
                line = self._format_feature_vector_line(
                    label=label,
                    qid=qid,
                    feats=feats,
                    external_id=eid
                )
                lines.append(line)

        # Write all training vectors.
        if train_feat_path is not None:
            Util.file_write_strings(train_feat_path, lines)

        # Call toolkit to train model.
        if self.toolkit == 'svmrank':
            self._train_svmrank(train_feat_path, self.model_file)
        else:
            self._train_ranklib(train_feat_path, self.model_file)

    @staticmethod
    def _avg_field_length(field):
        """Average length of a field."""
        doc_count = Idx.getDocCount(field)
        if doc_count == 0:
            return 0.0
        return Idx.getSumOfFieldLengths(field) / doc_count

    @staticmethod
    def _bm25_for_field(q_stems, docid, field, k1, b):
        """
        BM25 score for <q, d_field>, using term vectors.
        Returns None if the field is missing or empty.
        """
        tv = Idx.getTermVector(docid, field)
        if tv is None or tv.stemsLength() == 0 or tv.positionsLength() == 0:
            return None

        dl = Idx.getFieldLength(field, docid)
        if dl == 0:
            return None

        avgdl = RerankWithLtr._avg_field_length(field)
        N = Idx.getNumDocs()

        score = 0.0

        for stem in q_stems:
            idx = tv.indexOfStem(stem)
            if idx < 0:
                continue

            df = tv.stemDf(idx)
            tf = tv.stemFreq(idx)
            if df == 0 or tf <= 0:
                continue

            idf = math.log((N - df + 0.5) / (df + 0.5))
            denom = tf + k1 * ((1.0 - b) + b * (dl / avgdl))
            term_score = idf * (tf * (k1 + 1.0) / denom)
            score += term_score

        return score

    @staticmethod
    def _ql_dirichlet_for_field(q_stems, docid, field, mu):
        """
        Query Likelihood with Dirichlet smoothing for <q, d_field>.
        Returns a log-probability, or None if the field is missing/empty.
        """
        tv = Idx.getTermVector(docid, field)
        if tv is None or tv.stemsLength() == 0 or tv.positionsLength() == 0:
            return None

        dl = Idx.getFieldLength(field, docid)
        if dl == 0:
            return None

        collection_len = Idx.getSumOfFieldLengths(field)
        if collection_len == 0:
            return None

        score = 0.0
        seen = set()

        for stem in q_stems:
            if stem in seen:
                continue
            seen.add(stem)

            idx = tv.indexOfStem(stem)
            tf = tv.stemFreq(idx) if idx >= 0 else 0

            ctf = Idx.getTotalTermFreq(field, stem)
            if ctf == 0 and tf == 0:
                continue

            p_c = ctf / collection_len
            numerator = tf + mu * p_c
            denom = dl + mu
            term_prob = numerator / denom

            if term_prob > 0.0:
                score += math.log(term_prob)

        return score

    @staticmethod
    def _coord_match_for_field(q_stems, docid, field):
        """
        Coordinate Match (term overlap) for <q, d_field>.
        Returns None if the field is missing/empty.
        """
        tv = Idx.getTermVector(docid, field)
        if tv is None or tv.stemsLength() == 0 or tv.positionsLength() == 0:
            return None

        unique_q = set(q_stems)
        overlap = 0

        for stem in unique_q:
            if tv.indexOfStem(stem) >= 0:
                overlap += 1

        return float(overlap)

    @staticmethod
    def _f1_spam_score(docid):
        """f1: Spam score for d (float)."""
        val = Idx.getAttribute("spamScore", docid)
        if val is None:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    @staticmethod
    def _f2_url_depth(docid):
        """f2: URL depth (# of '/' in rawUrl)."""
        raw_url = Idx.getAttribute("rawUrl", docid)
        if raw_url is None:
            return None
        return float(raw_url.count('/'))

    @staticmethod
    def _f3_from_wikipedia(docid):
        """f3: 1 if rawUrl contains 'wikipedia.org', else 0."""
        raw_url = Idx.getAttribute("rawUrl", docid)
        if raw_url is None:
            return 0.0
        return 1.0 if "wikipedia.org" in raw_url.lower() else 0.0

    @staticmethod
    def _f4_pagerank(docid):
        """f4: PageRank score (float)."""
        val = Idx.getAttribute("PageRank", docid)
        if val is None:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    def _compute_standard_features(self, q_stems, docid):
        """
        Compute f1–f16 for <q, d>. Returns a dict {feature_id: value or None}.
        None means 'feature not applicable for this document'.
        """
        feats = {}

        # f1–f4: document-level attributes.
        feats[1] = self._f1_spam_score(docid)
        feats[2] = self._f2_url_depth(docid)
        feats[3] = self._f3_from_wikipedia(docid)
        feats[4] = self._f4_pagerank(docid)

        # f5–f7: body
        feats[5] = self._bm25_for_field(q_stems, docid, 'body',
                                        self.bm25_k1, self.bm25_b)
        feats[6] = self._ql_dirichlet_for_field(q_stems, docid, 'body',
                                                self.ql_mu)
        feats[7] = self._coord_match_for_field(q_stems, docid, 'body')

        # f8–f10: title
        feats[8] = self._bm25_for_field(q_stems, docid, 'title',
                                        self.bm25_k1, self.bm25_b)
        feats[9] = self._ql_dirichlet_for_field(q_stems, docid, 'title',
                                                self.ql_mu)
        feats[10] = self._coord_match_for_field(q_stems, docid, 'title')

        # f11–f13: url
        feats[11] = self._bm25_for_field(q_stems, docid, 'url',
                                         self.bm25_k1, self.bm25_b)
        feats[12] = self._ql_dirichlet_for_field(q_stems, docid, 'url',
                                                 self.ql_mu)
        feats[13] = self._coord_match_for_field(q_stems, docid, 'url')

        # f14–f16: inlink
        feats[14] = self._bm25_for_field(q_stems, docid, 'inlink',
                                         self.bm25_k1, self.bm25_b)
        feats[15] = self._ql_dirichlet_for_field(q_stems, docid, 'inlink',
                                                 self.ql_mu)
        feats[16] = self._coord_match_for_field(q_stems, docid, 'inlink')

        return feats

    def _apply_feature_disable(self, feats):
        """Remove globally disabled features from the feature dict."""
        if not self.disabled_features:
            return feats
        return {fid: v for fid, v in feats.items() if fid not in self.disabled_features}

    @staticmethod
    def _normalize_per_query(per_query_feats):
        """
        For SVMrank: normalize each feature dimension to [0,1] for a
        single query, as described in the design guide.
        Missing (None) features are set to 0 AFTER normalization.
        """
        if not per_query_feats:
            return per_query_feats

        # Gather all feature ids present across docs.
        all_fids = set()
        for feats in per_query_feats:
            all_fids.update(feats.keys())

        mins = {}
        maxs = {}

        for fid in all_fids:
            vals = [feats.get(fid) for feats in per_query_feats if feats.get(fid) is not None]
            if not vals:
                # All missing; handled later as zeros.
                continue
            mins[fid] = min(vals)
            maxs[fid] = max(vals)

        normed = []
        for feats in per_query_feats:
            new_feats = {}
            for fid in all_fids:
                v = feats.get(fid)
                if v is None:
                    # Missing; set to 0 after normalization.
                    new_feats[fid] = 0.0
                    continue

                if fid not in mins or fid not in maxs:
                    # No variation; treat as 0.
                    new_feats[fid] = 0.0
                    continue

                mn = mins[fid]
                mx = maxs[fid]
                if mx == mn:
                    new_feats[fid] = 0.0
                else:
                    new_feats[fid] = (v - mn) / (mx - mn)

            normed.append(new_feats)

        return normed

    @staticmethod
    def _fill_missing_with_zero(per_query_feats):
        """
        For RankLib: ensure that every feature dict has an explicit
        value (0.0) for every feature id seen for that query.
        """
        if not per_query_feats:
            return per_query_feats

        all_fids = set()
        for feats in per_query_feats:
            all_fids.update(feats.keys())

        filled = []
        for feats in per_query_feats:
            new_feats = {}
            for fid in all_fids:
                v = feats.get(fid)
                new_feats[fid] = 0.0 if v is None else v
            filled.append(new_feats)
        return filled

    @staticmethod
    def _format_feature_vector_line(label, qid, feats, external_id):
        """
        Convert a (label, qid, feature dict, externalId) into the
        toolkit feature-vector line format:

        <label> qid:<qid> 1:val1 2:val2 ... # externalId
        """
        # Features must be in canonical order by increasing feature id.
        parts = [str(int(label)) if float(label).is_integer() else str(float(label))]
        parts.append(f'qid:{qid}')

        for fid in sorted(feats.keys()):
            parts.append(f'{fid}:{feats[fid]}')

        line = ' '.join(parts) + f' # {external_id}'
        return line

    def _train_svmrank(self, train_feat_path, model_path):
        """
        Call SVMrank (external executable) to train a model.
        """
        learn_path = self.params.get('ltr:svmRankLearnPath')
        c_val = self.params.get('ltr:svmRankParamC', 0.001)
        if learn_path is None:
            return

        cmd = f'{learn_path} -c {c_val} {train_feat_path} {model_path}'
        subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            shell=True
        ).decode('UTF-8')

    def _train_ranklib(self, train_feat_path, model_path):
        """
        Call RankLib (via PyLu.RankLib) to train a model.
        """
        ranker = self.params.get('ltr:RankLib:model')
        metric2t = self.params.get('ltr:RankLib:metric2t')

        args = [
            '-train', train_feat_path,
            '-ranker', str(ranker),
            '-save', model_path
        ]
        if metric2t is not None:
            args.extend(['-metric2t', metric2t])

        PyLu.RankLib.main(args)

    def _score_test_vectors(self, testing_path, scores_path):
        """
        Use the trained model and the configured toolkit to score the
        testing feature vectors.
        """
        if testing_path is None or scores_path is None or self.model_file is None:
            return

        if self.toolkit == 'svmrank':
            classify_path = self.params.get('ltr:svmRankClassifyPath')
            if classify_path is None:
                return
            cmd = f'{classify_path} {testing_path} {self.model_file} {scores_path}'
            subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                shell=True
            ).decode('UTF-8')
        else:
            # RankLib scoring.
            args = [
                '-rank', testing_path,
                '-load', self.model_file,
                '-score', scores_path
            ]
            PyLu.RankLib.main(args)
