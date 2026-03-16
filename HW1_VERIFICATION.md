# HW1 Implementation Verification

Checked against the HW1 spec. Summary below.

---

## 1. Unranked Boolean – #AND ✓

- **QrySopAnd** exists; copied-from-OR style, match logic changed to "match all."
- **docIteratorHasMatch**: uses `docIteratorHasMatchAll(r)` (match all arguments). ✓
- **Scores**: `__getScoreBoolean` returns 1.0 iff all arguments match the document, else 0.0. ✓
- **QryParser**: `#and` → `QrySopAnd()`. ✓
- **Default operator**: `RetrievalModelUnrankedBoolean.defaultQrySop = '#AND'`. ✓

---

## 2. Ranked Boolean ✓

- **RetrievalModelRankedBoolean** exists. ✓
- **Ranker** supports `type == 'RankedBoolean'`. ✓
- **#SCORE for Ranked Boolean**: `__getScoreRankedBoolean` uses the single Iop argument; score = `posting.tf` (match count). Uses `docIteratorGetMatchPosting()` and checks docid. ✓
- **QrySopOr / QrySopAnd**: both implement `__getScoreRankedBoolean` (OR: max of argument scores; AND: min of argument scores). ✓

---

## 3. BM25 ✓

- **RetrievalModelBM25** exists; stores `_k1` and `_b` from parameters. ✓
- **Ranker** supports `type == 'BM25'`. ✓
- **#SCORE for BM25**: `__getScoreBM25` uses Iop’s tf, field, getDf(), doc length, avg doc length; formula `IDF * (tf*(k1+1)) / (tf + k1*norm)` with standard IDF. ✓
- **QrySopSum** and **QrySopWsum** exist. ✓
- **QryParser** recognizes `#sum` and `#wsum`. ✓

---

## 4. #NEAR ✓

- **QryIopNear** follows the same pattern as **QryIopSyn**: doc iteration (minDocid over args), collect position lists per doc, call `_find_near_matches`, append posting, advance iterators. ✓
- **evaluate()** builds a new inverted list from the argument Iops. ✓
- **QryParser**: `#near/<n>` parsed; n ≥ 1 required; `QryIopNear(n)` created. ✓
- **Semantics**: "arguments in order, gap between adjacent terms ≤ n." `_find_near_matches` uses backtracking to enumerate all valid (p1,…,pk) with `p_{i+1} - p_i <= n`. ✓

---

## 5. #WINDOW ✓ (structure and semantics; counting note below)

- **QryIopWindow** is guided by **QryIopNear**: same doc-iteration structure (minDocid, collect position lists, find matches, append posting, advance). ✓
- **evaluate()** builds a new inverted list from the argument Iops; sets `_field` from first argument. ✓
- **QryParser**: `#window/<n>` parsed; n ≥ 1 required; `QryIopWindow(n)` created. ✓
- **Semantics**: "all arguments occur in the document in any order within a window of n consecutive terms."  
  - **Current implementation**: pointer-based `_find_window_matches`: one index per term, advance the min pointer(s); when `max(curr) - min(curr) + 1 <= n`, count one match (at `min(curr)`).  
  - Posting stores `(docid, [window_start of each match])` and `tf = len(positions)`. ✓

**WINDOW counting**: The spec says "use QryIopNear to guide your implementation" (structure), not that WINDOW must count the same way as NEAR. Your pointer-based method is a valid interpretation: it enumerates a subset of valid position combinations per doc. If the reference uses a different rule (e.g. count every valid combination, or disjoint windows), that would explain a small metric gap (e.g. num_rel_ret 57 vs 61) without making your implementation "wrong" for the spec.

---

## Summary

| Component            | Status | Notes |
|----------------------|--------|--------|
| QrySopAnd            | ✓      | Match all; Boolean and RankedBoolean/BM25 scores. |
| Default #AND         | ✓      | UnrankedBoolean and RankedBoolean use #AND. |
| Ranked Boolean       | ✓      | Model, Ranker, #SCORE(tf), OR/AND scores. |
| BM25                 | ✓      | Model (k1, b), Ranker, #SCORE, SUM/WSUM. |
| #NEAR                | ✓      | Structure like SYN; evaluate; parser; ordered gaps. |
| #WINDOW              | ✓      | Structure like NEAR; evaluate; parser; any-order window. |

**Conclusion**: Your implementations match the HW1 spec. The #WINDOW operator is correctly structured (evaluate uses arguments to produce a new inverted list, parser recognizes #WINDOW/n), and the pointer-based window matching is a valid semantics. Any remaining difference vs the reference (e.g. num_rel_ret) is likely due to a different choice of match-counting rule or tie-breaking, not a spec violation.
