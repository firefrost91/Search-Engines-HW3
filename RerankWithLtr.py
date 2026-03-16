"""
Access and manage a feature-based learning-to-rank (Ltr) reranker.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import math
import subprocess
from functools import lru_cache

import PyLu				# Used to access RankLib
import Util				# Used to read and write files

from Idx import Idx
from QryParser import QryParser


class RerankWithLtr:
    """
    Access and manage a feature-based learning-to-rank (Ltr) reranker.
    """

    # All supported fields for field-based features
    _FIELDS = ['body', 'title', 'url', 'inlink']

    # Feature numbers corresponding to (field, type) pairs:
    # f5-f7: body BM25/QL/overlap, f8-f10: title, f11-f13: url, f14-f16: inlink
    _FIELD_FEATURES = {
        'body':   {'bm25': 5,  'ql': 6,  'overlap': 7},
        'title':  {'bm25': 8,  'ql': 9,  'overlap': 10},
        'url':    {'bm25': 11, 'ql': 12, 'overlap': 13},
        'inlink': {'bm25': 14, 'ql': 15, 'overlap': 16},
    }

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):

        # Store the parameters for the LTR reranker
        self._params = parameters
        self._toolkit = parameters.get('ltr:toolkit', 'RankLib').strip()

        # BM25 parameters for features
        self._bm25_k1 = float(parameters.get('ltr:BM25:k_1', 1.2))
        self._bm25_b  = float(parameters.get('ltr:BM25:b', 0.75))

        # QL (Dirichlet) mu parameter
        self._ql_mu = float(parameters.get('ltr:QL:mu', 2500.0))

        # Disabled features (set of ints)
        disabled_str = parameters.get('ltr:featureDisable', '')
        if disabled_str:
            self._disabled = set(int(x.strip()) for x in str(disabled_str).split(',') if x.strip())
        else:
            self._disabled = set()

        # Paths
        self._train_qry_file   = parameters.get('ltr:trainingQueryFile')
        self._train_qrels_file = parameters.get('ltr:trainingQrelsFile')
        self._train_fv_file    = parameters.get('ltr:trainingFeatureVectorsFile')
        self._test_fv_file     = parameters.get('ltr:testingFeatureVectorsFile')
        self._test_scores_file = parameters.get('ltr:testingDocumentScores')
        self._model_file       = parameters.get('ltr:modelFile')

        # Precompute collection-wide stats for each field (used by BM25/QL features)
        self._field_stats = {}
        for field in self._FIELDS:
            try:
                num_docs  = float(Idx.getDocCount(field))
                sum_len   = float(Idx.getSumOfFieldLengths(field))
                avg_len   = sum_len / num_docs if num_docs > 0 else 1.0
                self._field_stats[field] = {
                    'num_docs': num_docs,
                    'sum_len':  sum_len,
                    'avg_len':  avg_len,
                }
            except Exception:
                self._field_stats[field] = {'num_docs': 0.0, 'sum_len': 0.0, 'avg_len': 1.0}

        # Per-query caches (reset each query)
        self._query_cache = {}

        # Train a model
        self._train()


    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def _train(self):
        """Generate training feature vectors and train a model."""
        if not self._train_qry_file or not self._train_qrels_file:
            return

        # Read training queries
        train_queries = Util.read_queries(self._train_qry_file)

        # Read qrels: list of [qid, _, docid, rel]
        qrels_raw = Util.read_qrels(self._train_qrels_file)

        # Build qrels dict: {qid: {docid: rel}}
        qrels = {}
        for row in qrels_raw:
            if len(row) < 4:
                continue
            qid, _, docid, rel = row[0], row[1], row[2], int(row[3])
            # Treat -2 as 0
            if rel < 0:
                rel = 0
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = rel

        # Generate feature vectors for training queries
        all_fv_lines = []
        for qid in sorted(qrels.keys(), key=lambda x: int(x)):
            if qid not in train_queries:
                continue
            qstring = train_queries[qid]
            query_stems = QryParser.tokenizeString(qstring)

            self._reset_query_cache()

            # Feature vectors for this query (before normalization)
            fv_list = []  # list of (rel, docid, feature_dict)
            for docid_ext, rel in qrels[qid].items():
                try:
                    docid = Idx.getInternalDocid(docid_ext)
                except Exception:
                    continue
                fv = self._generate_feature_vector(qid, query_stems, docid, docid_ext)
                fv_list.append((rel, docid_ext, fv))

            # Normalize for SVMrank
            if self._toolkit == 'SVMRank':
                fv_list = self._normalize_fv_list(fv_list)

            # Format and collect lines
            for rel, docid_ext, fv in fv_list:
                line = self._format_fv_line(rel, qid, fv, docid_ext)
                all_fv_lines.append(line)

        # Write and flush training feature vectors
        with open(self._train_fv_file, 'w') as f:
            for line in all_fv_lines:
                f.write(line + '\n')
            f.flush()

        # Train the model
        self._call_toolkit_train()


    # ------------------------------------------------------------------ #
    # Reranking                                                            #
    # ------------------------------------------------------------------ #

    def rerank(self, batch):
        """
        Update the results for a set of queries with new scores.

        batch: A dict of {qid: {'qstring': qstring,
                                'ranking': [(score, externalId) ...]}
                          ... }
        """
        all_fv_lines = []
        # Track order for score reading: list of (qid, externalId)
        doc_order = []

        for qid in sorted(batch.keys(), key=lambda x: int(x)):
            qstring = batch[qid]['qstring']
            # Convert PRF query (with #SUM/#WSUM) to BOW
            qstring = QryParser.bowQuery(qstring)
            query_stems = QryParser.tokenizeString(qstring)

            self._reset_query_cache()

            ranking = batch[qid]['ranking']
            fv_list = []  # list of (0, docid_ext, fv)
            for score, docid_ext in ranking:
                try:
                    docid = Idx.getInternalDocid(docid_ext)
                except Exception:
                    docid = -1
                fv = self._generate_feature_vector(qid, query_stems, docid, docid_ext)
                fv_list.append((0, docid_ext, fv))
                doc_order.append((qid, docid_ext))

            # Normalize for SVMrank
            if self._toolkit == 'SVMRank':
                fv_list = self._normalize_fv_list(fv_list)

            for rel, docid_ext, fv in fv_list:
                line = self._format_fv_line(rel, qid, fv, docid_ext)
                all_fv_lines.append(line)

        # Write and flush test feature vectors
        with open(self._test_fv_file, 'w') as f:
            for line in all_fv_lines:
                f.write(line + '\n')
            f.flush()

        # Generate new scores
        self._call_toolkit_score()

        # Read new scores and re-rank
        new_scores = self._read_scores(doc_order)

        # Update batch rankings
        results = {}
        for qid in batch:
            new_ranking = []
            for score, docid_ext in batch[qid]['ranking']:
                key = (qid, docid_ext)
                new_score = new_scores.get(key, score)
                new_ranking.append((new_score, docid_ext))
            new_ranking.sort(key=lambda x: x[0], reverse=True)
            results[qid] = {'qstring': batch[qid]['qstring'], 'ranking': new_ranking}

        return results


    # ------------------------------------------------------------------ #
    # Feature generation                                                   #
    # ------------------------------------------------------------------ #

    def _generate_feature_vector(self, qid, query_stems, docid, docid_ext):
        """
        Generate a feature dict {feature_id: value} for a (query, doc) pair.
        Features not applicable to this doc are omitted (SVMrank sparse) or
        will be handled at write time (RankLib dense with 0).
        """
        fv = {}

        # f1: Spam score
        if 1 not in self._disabled:
            val = self._get_spam_score(docid)
            if val is not None:
                fv[1] = val

        # f2: URL depth
        if 2 not in self._disabled:
            val = self._get_url_depth(docid)
            if val is not None:
                fv[2] = val

        # f3: FromWikipedia
        if 3 not in self._disabled:
            val = self._get_wikipedia_score(docid)
            if val is not None:
                fv[3] = val

        # f4: PageRank
        if 4 not in self._disabled:
            val = self._get_pagerank(docid)
            if val is not None:
                fv[4] = val

        # f5-f16: field-based features
        for field, feat_map in self._FIELD_FEATURES.items():
            tv = self._get_term_vector(docid, field)
            # Skip empty/missing field (not even try to compute features)
            field_valid = (tv is not None and
                           tv.positionsLength() > 0 and
                           tv.stemsLength() > 0)

            if feat_map['bm25'] not in self._disabled:
                if field_valid:
                    fv[feat_map['bm25']] = self._feature_bm25(query_stems, tv, field)
                # else: omit

            if feat_map['ql'] not in self._disabled:
                if field_valid:
                    fv[feat_map['ql']] = self._feature_ql(query_stems, tv, field)

            if feat_map['overlap'] not in self._disabled:
                if field_valid:
                    fv[feat_map['overlap']] = self._feature_overlap(query_stems, tv)

        # f17-f20: custom features
        self._add_custom_features(fv, query_stems, docid, docid_ext)

        return fv


    # ------------------------------------------------------------------ #
    # Document attribute helpers                                           #
    # ------------------------------------------------------------------ #

    def _get_spam_score(self, docid):
        try:
            val = Idx.getAttribute('spamScore', docid)
            if val is None:
                return None
            return float(val)
        except Exception:
            return None

    def _get_url_depth(self, docid):
        try:
            raw_url = Idx.getAttribute('rawUrl', docid)
            if raw_url is None:
                return None
            return float(str(raw_url).count('/'))
        except Exception:
            return None

    def _get_wikipedia_score(self, docid):
        try:
            raw_url = Idx.getAttribute('rawUrl', docid)
            if raw_url is None:
                return None
            return 1.0 if 'wikipedia.org' in str(raw_url) else 0.0
        except Exception:
            return None

    def _get_pagerank(self, docid):
        try:
            val = Idx.getAttribute('PageRank', docid)
            if val is None:
                return None
            return float(val)
        except Exception:
            return None


    # ------------------------------------------------------------------ #
    # Term vector cache                                                    #
    # ------------------------------------------------------------------ #

    def _reset_query_cache(self):
        self._query_cache = {}

    def _get_term_vector(self, docid, field):
        key = ('tv', docid, field)
        if key not in self._query_cache:
            try:
                tv = Idx.getTermVector(docid, field)
                self._query_cache[key] = tv
            except Exception:
                self._query_cache[key] = None
        return self._query_cache[key]

    def _get_ctf(self, field, stem):
        key = ('ctf', field, stem)
        if key not in self._query_cache:
            try:
                self._query_cache[key] = float(Idx.getTotalTermFreq(field, stem))
            except Exception:
                self._query_cache[key] = 0.0
        return self._query_cache[key]


    # ------------------------------------------------------------------ #
    # BM25 feature                                                         #
    # ------------------------------------------------------------------ #

    def _feature_bm25(self, query_stems, tv, field):
        stats = self._field_stats[field]
        num_docs = stats['num_docs']
        avg_len  = stats['avg_len']
        doc_len  = float(tv.positionsLength())
        k1 = self._bm25_k1
        b  = self._bm25_b

        score = 0.0
        for stem in query_stems:
            stem_idx = tv.indexOfStem(stem)
            if stem_idx < 0:
                continue
            tf  = float(tv.stemFreq(stem_idx))
            df  = float(tv.stemDf(stem_idx))
            if df <= 0:
                continue
            rsj = max(0.0, math.log((num_docs - df + 0.5) / (df + 0.5)))
            tf_weight = tf / (tf + k1 * (1 - b + b * doc_len / avg_len))
            score += rsj * tf_weight

        return score


    # ------------------------------------------------------------------ #
    # Query Likelihood feature (Dirichlet smoothing)                       #
    # ------------------------------------------------------------------ #

    def _feature_ql(self, query_stems, tv, field):
        mu  = self._ql_mu
        stats = self._field_stats[field]
        sum_len = stats['sum_len']
        doc_len = float(tv.positionsLength())

        score = 0.0
        for stem in query_stems:
            stem_idx = tv.indexOfStem(stem)
            tf = float(tv.stemFreq(stem_idx)) if stem_idx >= 0 else 0.0
            ctf = self._get_ctf(field, stem)
            if sum_len <= 0 or ctf <= 0:
                continue
            p_ml_c = ctf / sum_len
            p_smooth = (tf + mu * p_ml_c) / (doc_len + mu)
            if p_smooth > 0:
                score += math.log(p_smooth)

        return score


    # ------------------------------------------------------------------ #
    # Coordinate Match (overlap) feature                                   #
    # ------------------------------------------------------------------ #

    def _feature_overlap(self, query_stems, tv):
        count = 0
        for stem in query_stems:
            if tv.indexOfStem(stem) >= 0:
                count += 1
        return float(count)


    # ------------------------------------------------------------------ #
    # Custom features f17-f20                                              #
    # ------------------------------------------------------------------ #

    def _add_custom_features(self, fv, query_stems, docid, docid_ext):
        """
        f17: Query coverage ratio for body (overlap / query length).
             Measures what fraction of query terms appear in the document body.
        f18: BM25 score normalized by document length (body).
             Penalizes very long documents, rewarding concise matches.
        f19: Term frequency sum in body (total number of query term occurrences).
             Captures raw term frequency signal beyond BM25's saturation.
        f20: URL contains query term (1 if any query stem appears in rawUrl, 0 otherwise).
             Captures navigational query intent.
        """
        # f17: Query coverage ratio
        if 17 not in self._disabled and query_stems:
            tv_body = self._get_term_vector(docid, 'body')
            if tv_body is not None and tv_body.positionsLength() > 0:
                matched = sum(1 for s in query_stems if tv_body.indexOfStem(s) >= 0)
                fv[17] = matched / len(query_stems)

        # f18: BM25 / log(1 + doc_length) for body
        if 18 not in self._disabled:
            tv_body = self._get_term_vector(docid, 'body')
            if tv_body is not None and tv_body.positionsLength() > 0:
                bm25_val = self._feature_bm25(query_stems, tv_body, 'body')
                doc_len = float(tv_body.positionsLength())
                fv[18] = bm25_val / math.log(1.0 + doc_len) if doc_len > 0 else 0.0

        # f19: Total TF sum in body
        if 19 not in self._disabled:
            tv_body = self._get_term_vector(docid, 'body')
            if tv_body is not None and tv_body.positionsLength() > 0:
                tf_sum = 0.0
                for stem in query_stems:
                    idx = tv_body.indexOfStem(stem)
                    if idx >= 0:
                        tf_sum += float(tv_body.stemFreq(idx))
                fv[19] = tf_sum

        # f20: URL contains query stem
        if 20 not in self._disabled:
            try:
                raw_url = Idx.getAttribute('rawUrl', docid)
                if raw_url is not None:
                    url_lower = str(raw_url).lower()
                    hit = any(s in url_lower for s in query_stems)
                    fv[20] = 1.0 if hit else 0.0
            except Exception:
                pass


    # ------------------------------------------------------------------ #
    # Normalization                                                         #
    # ------------------------------------------------------------------ #

    def _normalize_fv_list(self, fv_list):
        """
        Normalize all feature values for a query to [0, 1].
        Missing features remain missing (omitted after normalization,
        not set to 0 before).
        """
        if not fv_list:
            return fv_list

        # Collect all feature ids present
        all_fids = set()
        for _, _, fv in fv_list:
            all_fids.update(fv.keys())

        # Find min/max for each feature id
        min_vals = {}
        max_vals = {}
        for fid in all_fids:
            vals = [fv[fid] for _, _, fv in fv_list if fid in fv]
            if vals:
                min_vals[fid] = min(vals)
                max_vals[fid] = max(vals)

        # Normalize
        normalized = []
        for rel, docid_ext, fv in fv_list:
            new_fv = {}
            for fid, val in fv.items():
                mn = min_vals[fid]
                mx = max_vals[fid]
                if mx == mn:
                    new_fv[fid] = 0.0
                else:
                    new_fv[fid] = (val - mn) / (mx - mn)
            normalized.append((rel, docid_ext, new_fv))

        return normalized


    # ------------------------------------------------------------------ #
    # File formatting                                                       #
    # ------------------------------------------------------------------ #

    def _format_fv_line(self, rel, qid, fv, docid_ext):
        """
        Format one feature vector line in SVMrank/RankLib format.
        For RankLib, all features 1..max must be present (dense).
        For SVMrank, sparse representation is allowed.
        """
        # Determine the max feature id to support
        max_fid = 20

        parts = [str(rel), f'qid:{qid}']

        if self._toolkit == 'RankLib':
            # Dense: include all feature ids 1..max_fid
            for fid in range(1, max_fid + 1):
                if fid in self._disabled:
                    continue
                val = fv.get(fid, 0.0)
                parts.append(f'{fid}:{val}')
        else:
            # Sparse SVMrank: only include present features in order
            for fid in sorted(fv.keys()):
                parts.append(f'{fid}:{fv[fid]}')

        parts.append(f'# {docid_ext}')
        return ' '.join(parts)


    # ------------------------------------------------------------------ #
    # Toolkit calls                                                         #
    # ------------------------------------------------------------------ #

    def _call_toolkit_train(self):
        """Train a model using the configured toolkit."""
        if self._toolkit == 'RankLib':
            ranklib_args = [
                '-train',   self._train_fv_file,
                '-ranker',  str(int(self._params.get('ltr:RankLib:model', 4))),
                '-save',    self._model_file,
            ]
            metric2t = self._params.get('ltr:RankLib:metric2t')
            if metric2t:
                ranklib_args += ['-metric2t', str(metric2t)]
            PyLu.RankLib.main(ranklib_args)

        elif self._toolkit == 'SVMRank':
            c = self._params.get('ltr:svmRankParamC', 0.001)
            learn_path = self._params['ltr:svmRankLearnPath']
            cmd = f'{learn_path} -c {c} {self._train_fv_file} {self._model_file}'
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode('UTF-8')
            print(output)


    def _call_toolkit_score(self):
        """Generate scores for test feature vectors."""
        if self._toolkit == 'RankLib':
            PyLu.RankLib.main([
                '-rank',  self._test_fv_file,
                '-load',  self._model_file,
                '-score', self._test_scores_file,
            ])

        elif self._toolkit == 'SVMRank':
            classify_path = self._params['ltr:svmRankClassifyPath']
            cmd = f'{classify_path} {self._test_fv_file} {self._model_file} {self._test_scores_file}'
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True).decode('UTF-8')
            print(output)


    # ------------------------------------------------------------------ #
    # Score reading                                                         #
    # ------------------------------------------------------------------ #

    def _read_scores(self, doc_order):
        """
        Read new scores from the toolkit output file.
        Returns a dict {(qid, docid_ext): score}.
        """
        lines = Util.file_read_strings(self._test_scores_file)
        scores = {}

        if self._toolkit == 'SVMRank':
            # One score per line, same order as doc_order
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                if i < len(doc_order):
                    qid, docid_ext = doc_order[i]
                    scores[(qid, docid_ext)] = float(line)

        elif self._toolkit == 'RankLib':
            # Format: qid<tab>rank<tab>score
            # Same order as doc_order
            idx = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                score = float(parts[2])
                if idx < len(doc_order):
                    qid, docid_ext = doc_order[idx]
                    scores[(qid, docid_ext)] = score
                idx += 1

        return scores
