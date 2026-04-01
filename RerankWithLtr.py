"""
Access and manage a feature-based learning-to-rank (Ltr) reranker.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import math
import subprocess

import PyLu                         # Used to access RankLib
import Util                         # Used to read and write files

from Idx import Idx
from QryParser import QryParser     # Parse queries


class RerankWithLtr:
    """
    Access and manage a feature-based learning-to-rank (Ltr) reranker.

    Features:
      f1  - Spam score (percentile; higher = less spam)
      f2  - URL depth (number of '/' in rawUrl)
      f3  - FromWikipedia (1 if rawUrl contains "wikipedia.org", else 0)
      f4  - PageRank score
      f5  - BM25 score for <q, d_body>
      f6  - Query Likelihood (Dirichlet) for <q, d_body>  [Indri AND]
      f7  - Term overlap (Coordinate Match) for <q, d_body>
      f8  - BM25 score for <q, d_title>
      f9  - Query Likelihood (Dirichlet) for <q, d_title> [Indri AND]
      f10 - Term overlap for <q, d_title>
      f11 - BM25 score for <q, d_url>
      f12 - Query Likelihood (Dirichlet) for <q, d_url>   [Indri AND]
      f13 - Term overlap for <q, d_url>
      f14 - BM25 score for <q, d_inlink>
      f15 - Query Likelihood (Dirichlet) for <q, d_inlink>[Indri AND]
      f16 - Term overlap for <q, d_inlink>
      f17 - IDF-weighted body coverage:
            fraction of query-term weight matched in body
      f18 - Query-window density in body:
            rewards query terms that occur close together
      f19 - Reciprocal first-hit position in title:
            rewards title matches that happen early
      f20 - IDF-weighted cross-field support:
            rewards query terms corroborated across body/title/url/inlink/keywords

    Caching strategy (design guide efficiency requirement):
      - Corpus-wide constants (N, sum_len, avg_len per field) are computed
        once in __init__ and reused for every query.
      - Per-query caches (_tv_cache, _idf_cache, _ctf_cache) are dicts
        that are reset to {} at the start of each query.  Each lookup
        checks the cache before touching the index.  get_xxx helpers
        encapsulate the check-then-fetch pattern so callers never have
        to think about it.
    """

    # Fields that have BM25/QL/overlap features (f5-f16).
    _SCORED_FIELDS = ['body', 'title', 'url', 'inlink']

    # All fields for which we need corpus-wide stats.
    _ALL_FIELDS = ['body', 'title', 'url', 'inlink', 'keywords']

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):

        # ---- Store parameters ----
        self._params = parameters

        # Which features to disable (optional parameter).
        self._disabled = set()
        if 'ltr:featureDisable' in parameters:
            for s in str(parameters['ltr:featureDisable']).split(','):
                s = s.strip()
                if s.isdigit():
                    self._disabled.add(int(s))

        self._toolkit  = str(parameters.get('ltr:toolkit', 'RankLib'))
        self._bm25_k1  = float(parameters.get('ltr:BM25:k_1', 1.2))
        self._bm25_b   = float(parameters.get('ltr:BM25:b',   0.75))
        self._ql_mu    = float(parameters.get('ltr:QL:mu',    2500.0))

        # ---- Corpus-wide constants (computed once, reused every query) ----
        # These never change across queries, so we cache them permanently.
        self._N       = {}   # {field: doc count}
        self._sum_len = {}   # {field: sum of field lengths across corpus}
        self._avg_len = {}   # {field: average field length}

        for field in self._ALL_FIELDS:
            n = float(Idx.getDocCount(field))
            s = float(Idx.getSumOfFieldLengths(field))
            self._N[field]       = n
            self._sum_len[field] = s
            self._avg_len[field] = (s / n) if n > 0 else 1.0

        # ---- Per-query caches (reset at the start of each query) ----
        self._tv_cache  = {}  # (docid, field)  -> TermVector or None
        self._idf_cache = {}  # (field, stem)   -> BM25 RSJ IDF value
        self._ctf_cache = {}  # (field, stem)   -> collection term freq

        # ---- Build training data and train model ----
        self._train()


    # -------------- Cache helpers (get_xxx pattern) -------- #

    def _reset_query_caches(self):
        """Reset all per-query caches to empty dicts."""
        self._tv_cache  = {}
        self._idf_cache = {}
        self._ctf_cache = {}


    def _get_term_vector(self, docid, field):
        """
        Return the TermVector for (docid, field), using the per-query cache.
        Returns None if the field is empty or does not exist for this document.
        """
        key = (docid, field)
        if key not in self._tv_cache:
            tv = Idx.getTermVector(docid, field)
            # An empty TermVector has stemsLength() == 0 or positionsLength() == 0.
            if tv is None or tv.stemsLength() == 0 or tv.positionsLength() == 0:
                self._tv_cache[key] = None
            else:
                self._tv_cache[key] = tv
        return self._tv_cache[key]


    def _get_idf(self, field, stem):
        """
        Return the BM25 RSJ IDF for (field, stem), using the per-query cache.
        IDF = log((N - df + 0.5) / (df + 0.5))
        """
        key = (field, stem)
        if key not in self._idf_cache:
            df  = float(Idx.getDocFreq(field, stem))
            N   = self._N.get(field, 1.0)
            self._idf_cache[key] = max(0.0, math.log((N - df + 0.5) / (df + 0.5)))
        return self._idf_cache[key]


    def _get_ctf(self, field, stem):
        """
        Return the collection term frequency for (field, stem),
        using the per-query cache.
        """
        key = (field, stem)
        if key not in self._ctf_cache:
            self._ctf_cache[key] = float(Idx.getTotalTermFreq(field, stem))
        return self._ctf_cache[key]


    def _get_stem_tf(self, tv, stem):
        """
        Return the TF of stem in the term vector, or 0.0 if absent.
        Uses the TermVector's own index — no additional cache needed
        since indexOfStem is O(1) in the Java object.
        """
        idx = tv.indexOfStem(stem)
        if idx < 0:
            return 0.0
        tf = tv.stemFreq(idx)
        return float(tf) if tf > 0 else 0.0


    # -------------- Feature computation ------------------- #

    def _feature_bm25(self, query_stems, tv, field, docid):
        """
        BM25 score for <query_stems, tv> in the given field.
        Returns None if tv is None (field missing/empty for this doc).
        Only accumulates scores for query terms present in tv (BOW #SUM).
        """
        if tv is None:
            return None
        doc_len = float(Idx.getFieldLength(field, docid))
        if doc_len <= 0.0:
            return None

        avg_len = self._avg_len.get(field, 1.0)
        k1      = self._bm25_k1
        b       = self._bm25_b
        score   = 0.0

        for stem in query_stems:
            tf = self._get_stem_tf(tv, stem)
            if tf > 0.0:
                idf     = self._get_idf(field, stem)
                tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * doc_len / avg_len))
                score  += idf * tf_norm

        return score


    def _feature_ql(self, query_stems, tv, field, docid):
        """
        Query Likelihood score using Dirichlet smoothing and an Indri-style
        AND aggregation.
        Returns None if tv is None (field missing/empty for this doc).

        Per query term t:
            p(t|d) = (tf(t,d) + mu * p(t|C)) / (docLen + mu)
            where p(t|C) = ctf(t) / |C|

        Aggregate with geometric mean across query terms (Indri-style #AND):
            score = exp( (1/|q|) * sum_t log(p(t|d)) )

        Query terms that do not appear in the collection are skipped.
        """
        if tv is None:
            return None
        doc_len = float(Idx.getFieldLength(field, docid))
        if doc_len <= 0.0:
            return None

        mu      = self._ql_mu
        sum_len = self._sum_len.get(field, 0.0)
        if sum_len <= 0:
            return None

        log_sum = 0.0
        used_terms = 0
        for stem in query_stems:
            tf = self._get_stem_tf(tv, stem)
            ctf = self._get_ctf(field, stem)
            if ctf <= 0.0:
                continue
            p_tc = ctf / sum_len
            p_t_d = (tf + mu * p_tc) / (doc_len + mu)
            if p_t_d <= 0.0:
                continue
            log_sum += math.log(p_t_d)
            used_terms += 1

        if used_terms == 0:
            return 0.0

        return math.exp(log_sum / float(used_terms))


    def _feature_overlap(self, query_stems, tv):
        """
        Term overlap (Coordinate Match): count of query terms present in tv.
        Returns None if tv is None (field missing/empty for this doc).
        """
        if tv is None:
            return None
        count = 0
        for stem in query_stems:
            if tv.indexOfStem(stem) >= 0:
                count += 1
        return float(count)


    def _unique_query_terms(self, query_stems):
        """
        Return query terms with duplicates removed, preserving order.
        """
        return list(dict.fromkeys(query_stems))


    def _feature_weighted_coverage(self, query_stems, tv, field):
        """
        IDF-weighted fraction of unique query terms matched in a field.

        Weight each query term by (1 + idf) so the score remains defined
        even when a term's RSJ idf is 0.
        """
        if tv is None:
            return None

        query_terms = self._unique_query_terms(query_stems)
        if not query_terms:
            return 0.0

        matched_weight = 0.0
        total_weight = 0.0

        for stem in query_terms:
            weight = 1.0 + self._get_idf(field, stem)
            total_weight += weight
            if tv.indexOfStem(stem) >= 0:
                matched_weight += weight

        if total_weight <= 0.0:
            return 0.0

        return matched_weight / total_weight


    def _feature_window_density(self, query_stems, tv):
        """
        Query-term compactness in body.

        For each k from 2..|q|, find the tightest span containing at least k
        distinct query terms and score it as:

            (k / |q|) * (k / window_width)

        This rewards windows that cover more of the query in fewer tokens.
        """
        if tv is None:
            return None

        query_terms = self._unique_query_terms(query_stems)
        num_query_terms = len(query_terms)
        if num_query_terms < 2:
            return 0.0

        # Pre-build a map from stem index → stem string for only the query
        # terms that actually appear in this term vector.  This avoids calling
        # the expensive tv.stemString() Java bridge for every document position
        # in the inner loop (web docs can have 50k–100k+ positions).
        query_idx_to_stem = {}
        for stem in query_terms:
            idx = tv.indexOfStem(stem)
            if idx >= 0:
                query_idx_to_stem[idx] = stem

        occurrences = []
        if query_idx_to_stem:
            # Bulk-copy the Java int array to a Python list in one call so the
            # inner loop uses fast Python indexing instead of repeated Java
            # bridge calls (one per position).
            py_positions = list(tv.positions)
            for pos, stem_i in enumerate(py_positions):
                if stem_i in query_idx_to_stem:
                    occurrences.append((pos, query_idx_to_stem[stem_i]))

        if len(occurrences) < 2:
            return 0.0

        best = 0.0
        for need in range(2, num_query_terms + 1):
            counts = {}
            have = 0
            left = 0
            min_width = None

            for right, (rpos, rterm) in enumerate(occurrences):
                counts[rterm] = counts.get(rterm, 0) + 1
                if counts[rterm] == 1:
                    have += 1

                while have >= need and left <= right:
                    lpos, lterm = occurrences[left]
                    width = rpos - lpos + 1
                    if min_width is None or width < min_width:
                        min_width = width

                    counts[lterm] -= 1
                    if counts[lterm] == 0:
                        del counts[lterm]
                        have -= 1
                    left += 1

            if min_width is not None:
                coverage = float(need) / float(num_query_terms)
                density = float(need) / float(min_width)
                best = max(best, coverage * density)

        return best


    def _feature_first_hit(self, query_stems, tv):
        """
        Reciprocal of the earliest matching query-term position in a field.

        A match at position 0 gets score 1.0, position 1 gets 0.5, etc.
        """
        if tv is None:
            return None

        if not query_stems:
            return 0.0

        # Pre-build stem-index set for query terms present in this TV.
        query_indices = set()
        for stem in query_stems:
            idx = tv.indexOfStem(stem)
            if idx >= 0:
                query_indices.add(idx)

        if not query_indices:
            return 0.0

        py_positions = list(tv.positions)
        for pos, stem_i in enumerate(py_positions):
            if stem_i in query_indices:
                return 1.0 / float(1 + pos)

        return 0.0


    def _feature_cross_field_support(self, query_stems, docid):
        """
        IDF-weighted average fraction of fields that support each query term.

        Query terms that appear across multiple fields receive more credit
        than terms that appear in only one location.
        """
        query_terms = self._unique_query_terms(query_stems)
        if not query_terms:
            return 0.0

        total_weight = 0.0
        support = 0.0

        for stem in query_terms:
            weight = 1.0 + self._get_idf('body', stem)
            total_weight += weight

            matched_fields = 0
            for field in self._ALL_FIELDS:
                tv = self._get_term_vector(docid, field)
                if tv is not None and tv.indexOfStem(stem) >= 0:
                    matched_fields += 1

            support += weight * (float(matched_fields) / float(len(self._ALL_FIELDS)))

        if total_weight <= 0.0:
            return 0.0

        return support / total_weight


    def _generate_feature_vector(self, docid, query_stems):
        """
        Generate a feature vector for (docid, query_stems).

        Returns a dict {feature_id: value}, where value is a float or None.
        None means the feature does not apply to this <q,d> pair (e.g., the
        document has no inlink field).  None values must NOT be replaced with
        0 until after per-query normalization.
        """
        fv = {}

        # -- f1: Spam score --
        if 1 not in self._disabled:
            val = Idx.getAttribute('spamScore', docid)
            fv[1] = float(val) if val is not None else None

        # -- f2: URL depth --
        if 2 not in self._disabled:
            raw_url = Idx.getAttribute('rawUrl', docid)
            fv[2] = float(raw_url.count('/')) if raw_url is not None else None

        # -- f3: FromWikipedia --
        if 3 not in self._disabled:
            raw_url = Idx.getAttribute('rawUrl', docid)
            fv[3] = 1.0 if (raw_url is not None and 'wikipedia.org' in raw_url) else 0.0

        # -- f4: PageRank --
        if 4 not in self._disabled:
            val = Idx.getAttribute('PageRank', docid)
            fv[4] = float(val) if val is not None else None

        # -- f5-f16: BM25 / QL / Overlap for body, title, url, inlink --
        field_base = {'body': 5, 'title': 8, 'url': 11, 'inlink': 14}
        for field, base in field_base.items():
            f_bm25 = base
            f_ql   = base + 1
            f_ol   = base + 2

            # Fetch term vector once per (docid, field) per query via cache.
            tv = self._get_term_vector(docid, field)

            if f_bm25 not in self._disabled:
                fv[f_bm25] = self._feature_bm25(query_stems, tv, field, docid)
            if f_ql not in self._disabled:
                fv[f_ql]   = self._feature_ql(query_stems, tv, field, docid)
            if f_ol not in self._disabled:
                fv[f_ol]   = self._feature_overlap(query_stems, tv)

        tv_body = self._get_term_vector(docid, 'body')
        tv_title = self._get_term_vector(docid, 'title')

        # -- f17: IDF-weighted body coverage --
        if 17 not in self._disabled:
            fv[17] = self._feature_weighted_coverage(query_stems, tv_body, 'body')

        # -- f18: query-window density in body --
        if 18 not in self._disabled:
            fv[18] = self._feature_window_density(query_stems, tv_body)

        # -- f19: reciprocal first-hit position in title --
        if 19 not in self._disabled:
            fv[19] = self._feature_first_hit(query_stems, tv_title)

        # -- f20: IDF-weighted cross-field support --
        if 20 not in self._disabled:
            fv[20] = self._feature_cross_field_support(query_stems, docid)

        return fv


    # -------------- Normalization (SVMrank only) ---------- #

    def _normalize_feature_vectors(self, fv_list):
        """
        Normalize each feature to [0, 1] across all documents for one query
        (required for SVMrank).

        Rules (per design guide):
          - Use only non-None values to find min and max.
          - If min == max, set normalized value to 0.
          - Set None values to 0 AFTER normalization (not before), so they
            do not distort the min/max calculation.
        Modifies fv_list in-place.
        """
        # Collect all feature ids present in any vector.
        all_fids = set()
        for fv in fv_list:
            all_fids.update(fv.keys())

        for fid in all_fids:
            vals = [fv[fid] for fv in fv_list if fv.get(fid) is not None]

            if not vals:
                # Every doc is missing this feature — set all to 0.
                for fv in fv_list:
                    if fid in fv:
                        fv[fid] = 0.0
                continue

            mn = min(vals)
            mx = max(vals)

            for fv in fv_list:
                v = fv.get(fid)
                if v is None:
                    fv[fid] = 0.0           # missing → 0 after normalization
                elif mx == mn:
                    fv[fid] = 0.0           # constant feature → 0
                else:
                    fv[fid] = (v - mn) / (mx - mn)


    def _replace_none_with_zero(self, fv_list):
        """
        Replace all None feature values with 0.0.
        Used for RankLib (complete representation required).
        """
        for fv in fv_list:
            for fid in fv:
                if fv[fid] is None:
                    fv[fid] = 0.0


    # -------------- File I/O ------------------------------ #

    def _fv_to_line(self, rel, qid, fv, ext_id):
        """
        Render a feature vector as one line of the SVM-light feature file.

        Format: <rel> qid:<qid> <fid>:<val> ... # <ext_id>
        Features are written in ascending canonical order.
        """
        parts = [
            f'{fid}:{fv[fid]}'
            for fid in sorted(fv.keys())
        ]
        return f'{rel} qid:{qid} {" ".join(parts)} # {ext_id}'


    def _write_feature_vectors(self, path, records):
        """
        Write a list of (rel, qid, fv, ext_id) records to path.

        Files are explicitly closed (not just flushed) before the toolkit
        reads them — per the design guide warning about Python buffering.
        """
        with open(path, 'w') as f:
            for rel, qid, fv, ext_id in records:
                f.write(self._fv_to_line(rel, qid, fv, ext_id) + '\n')
        # Context manager guarantees close() even on exception.


    # -------------- Toolkit calls ------------------------- #

    def _call_toolkit_train(self, train_fv_path, model_path):
        """Train a model using the configured toolkit."""
        if self._toolkit.lower() == 'svmrank':
            learn_path = self._params['ltr:svmRankLearnPath']
            c          = self._params.get('ltr:svmRankParamC', 0.001)
            cmd = (f'{learn_path} -c {c} '
                   f'{train_fv_path} {model_path}')
            out = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, shell=True).decode('UTF-8')
            print(out)
        else:
            # RankLib — called directly (more efficient, fewer file problems).
            args = [
                '-train',  train_fv_path,
                '-ranker', str(self._params.get('ltr:RankLib:model', 4)),
                '-save',   model_path,
            ]
            metric = self._params.get('ltr:RankLib:metric2t', 'MAP')
            args += ['-metric2t', str(metric)]
            PyLu.RankLib.main(args)


    def _call_toolkit_score(self, test_fv_path, model_path, scores_path):
        """Generate new scores for test data using the trained model."""
        if self._toolkit.lower() == 'svmrank':
            classify_path = self._params['ltr:svmRankClassifyPath']
            cmd = (f'{classify_path} '
                   f'{test_fv_path} {model_path} {scores_path}')
            out = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, shell=True).decode('UTF-8')
            print(out)
        else:
            PyLu.RankLib.main([
                '-rank',  test_fv_path,
                '-load',  model_path,
                '-score', scores_path,
            ])


    def _read_new_scores(self, scores_path, records):
        """
        Read toolkit output scores and pair them with (qid, ext_id) from
        records (same order as the feature vector file).

        Returns a list of (qid, ext_id, score) tuples.

        SVMrank: one score per line (plain float).
        RankLib: tab-separated <qid>\\t<rank>\\t<score> per line.
        """
        lines = Util.file_read_strings(scores_path)
        results = []
        rec_idx = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if self._toolkit.lower() == 'svmrank':
                score = float(line)
            else:
                parts = line.split('\t')
                score = float(parts[2])
            _, qid, _, ext_id = records[rec_idx]
            results.append((qid, ext_id, score))
            rec_idx += 1
        return results


    # -------------- Training ------------------------------ #

    def _train(self):
        """
        Generate training feature vectors and train the LTR model.

        Steps (per design guide architecture):
          1. Read training queries and qrels.
          2. For each training query, reset per-query caches, then generate
             a feature vector for every <q, d> pair in the qrels.
          3. Normalize (SVMrank only) or zero-fill (RankLib).
          4. Write feature vectors to file.
          5. Call toolkit to train model.
        """
        train_qry_path = self._params['ltr:trainingQueryFile']
        train_qrels_path = self._params['ltr:trainingQrelsFile']
        train_fv_path  = self._params['ltr:trainingFeatureVectorsFile']
        model_path     = self._params['ltr:modelFile']

        # ---- Load training queries ----
        print('[LTR] Loading training queries ...', flush=True)
        train_queries = Util.read_queries(train_qry_path)

        # ---- Load qrels: {qid: {ext_id: rel}} ----
        # Column layout: qid  dummy  ext_id  rel
        # Relevance label -2 (spam) is treated as 0 per design guide.
        print('[LTR] Loading qrels ...', flush=True)
        qrels = {}
        for parts in Util.read_qrels(train_qrels_path):
            if len(parts) < 4:
                continue
            qid, ext_id = parts[0], parts[2]
            rel = int(float(parts[3].strip()))
            if rel < 0:
                rel = 0
            qrels.setdefault(qid, {})[ext_id] = rel

        # ---- Generate feature vectors for every training <q, d> ----
        records = []   # final list of (rel, qid, fv, ext_id)
        qids = sorted(qrels.keys(), key=lambda x: int(x) if x.isdigit() else x)
        print(f'[LTR] Generating training feature vectors for {len(qids)} queries ...', flush=True)

        for qi, qid in enumerate(qids):
            if qid not in train_queries:
                continue

            query_stems = QryParser.tokenizeString(train_queries[qid])
            if not query_stems:
                continue

            print(f'[LTR]   query {qi+1}/{len(qids)} qid={qid} ({len(qrels[qid])} docs) ...', flush=True)
            self._reset_query_caches()
            fv_list  = []
            doc_list = []   # (rel, ext_id)

            for ext_id, rel in qrels[qid].items():
                try:
                    docid = Idx.getInternalDocid(ext_id)
                except Exception:
                    continue
                fv = self._generate_feature_vector(docid, query_stems)
                fv_list.append(fv)
                doc_list.append((rel, ext_id))

            if not fv_list:
                continue

            # Normalize or zero-fill (per toolkit requirement).
            if self._toolkit.lower() == 'svmrank':
                self._normalize_feature_vectors(fv_list)
            else:
                self._replace_none_with_zero(fv_list)

            for i, (rel, ext_id) in enumerate(doc_list):
                records.append((rel, qid, fv_list[i], ext_id))

        # ---- Write training feature vectors ----
        print(f'[LTR] Writing {len(records)} training feature vectors to {train_fv_path} ...', flush=True)
        self._write_feature_vectors(train_fv_path, records)

        # ---- Train model ----
        print(f'[LTR] Training model (toolkit={self._toolkit}, model={self._params.get("ltr:RankLib:model","?")}) ...', flush=True)
        self._call_toolkit_train(train_fv_path, model_path)
        print('[LTR] Training complete.', flush=True)


    # -------------- Reranking ----------------------------- #

    def _tokenize_bow(self, qstring):
        """
        Convert qstring to a BOW query, then tokenize to stop/stemmed terms.
        """
        bow = QryParser.bowQuery(qstring)
        return QryParser.tokenizeString(bow)


    def rerank(self, batch):
        """
        Update the results for a set of queries with new scores.

        batch: A dict of {qid: {'qstring': qstring,
                                'ranking': [(score, externalId) ...]}
                          ... }

        Steps (per design guide architecture):
          1. For each query convert to BOW (handles PRF pipeline).
          2. Reset per-query caches; generate feature vectors for top-n docs.
          3. Normalize (SVMrank) or zero-fill (RankLib).
          4. Write feature vectors to file.
          5. Call toolkit to produce new scores.
          6. Read scores and rebuild rankings.
        """
        test_fv_path = self._params['ltr:testingFeatureVectorsFile']
        scores_path  = self._params['ltr:testingDocumentScores']
        model_path   = self._params['ltr:modelFile']

        # Sort by numeric qid for canonical feature vector file order.
        sorted_qids = sorted(batch.keys(),
                             key=lambda x: int(x) if x.isdigit() else x)

        records = []   # (rel=0, qid, fv, ext_id) in file order

        for qid in sorted_qids:
            query_stems = self._tokenize_bow(batch[qid]['qstring'])
            if not query_stems:
                continue

            self._reset_query_caches()
            fv_list  = []
            doc_list = []   # ext_id

            for _score, ext_id in batch[qid]['ranking']:
                try:
                    docid = Idx.getInternalDocid(ext_id)
                except Exception:
                    continue
                fv = self._generate_feature_vector(docid, query_stems)
                fv_list.append(fv)
                doc_list.append(ext_id)

            if not fv_list:
                continue

            # Normalize or zero-fill.
            if self._toolkit.lower() == 'svmrank':
                self._normalize_feature_vectors(fv_list)
            else:
                self._replace_none_with_zero(fv_list)

            for i, ext_id in enumerate(doc_list):
                records.append((0, qid, fv_list[i], ext_id))

        # ---- Write test feature vectors ----
        self._write_feature_vectors(test_fv_path, records)

        # ---- Score with toolkit ----
        self._call_toolkit_score(test_fv_path, model_path, scores_path)

        # ---- Read new scores and rebuild rankings ----
        scored = self._read_new_scores(scores_path, records)

        new_rankings = {}
        for qid, ext_id, score in scored:
            new_rankings.setdefault(qid, []).append((score, ext_id))

        # Sort each new ranking by descending score.
        for qid in new_rankings:
            new_rankings[qid].sort(key=lambda x: -x[0])

        # Build results dict; preserve original batch data for queries with
        # no new ranking (e.g., all docs failed getInternalDocid).
        results = {qid: dict(batch[qid]) for qid in batch}
        for qid, ranking in new_rankings.items():
            results[qid]['ranking'] = ranking

        return results
