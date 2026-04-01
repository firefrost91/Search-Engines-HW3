# Exp-3.1 Reranking Configurations

## Overview

This document describes the reranking configurations used in the Exp-3.1 experiments, including pipeline structure, parameter choices, and the intuitions or hypotheses motivating each configuration.

---

## Exp-3.1a — BM25 → BERT (L-6) → BERT (L-12)

**Pipeline:**
1. BM25 retrieval (k₁=1.5, b=0.50, top 1000)
2. MiniLM-L-6-v2 maxp rerank (rerankDepth=250, psgLen=150, psgStride=125, psgCnt=3)
3. MiniLM-L-12-v2 maxp rerank (rerankDepth=100, psgLen=150, psgStride=125, psgCnt=3)

**Intuition:** This serves as a cascaded BERT-only reranking baseline over BM25 retrieval. The hypothesis was that a lightweight 6-layer model could efficiently prune the initial BM25 list from 1000 to 250, reducing the computational cost for the heavier 12-layer model, which then focuses on top precision. The cascade allows applying more powerful models only where they matter most — at the top of the ranking.

**Results:** MAP=0.3815, NDCG@20=0.6256, P@5=0.4833

---

## Exp-3.1b — inRankFile → LTR → BERT (L-12)

**Pipeline:**
1. Pre-existing ranked list (inRankFile)
2. LTR rerank (RankLib model 4, NDCG@10 metric, rerankDepth=300, BM25: k₁=1.2 b=0.75, QL: mu=2500, featureDisable=20)
3. MiniLM-L-12-v2 maxp rerank (rerankDepth=100, psgLen=150, psgStride=125, psgCnt=3)

**Intuition:** This configuration combines two complementary signal types. LTR leverages hand-crafted features — PageRank, spam score, Wikipedia link counts, BM25/QL scores across title/body/anchor/inlink fields, window density, and cross-field support — which capture document quality and structural relevance. BERT then captures deep semantic relevance that LTR's sparse features cannot. The hypothesis was that these approaches are largely orthogonal: LTR filters based on authority and quality signals, while BERT refines based on contextual semantic matching.

**Results:** MAP=0.3885, NDCG@20=0.6361, P@5=0.5000 — **best overall**

---

## Exp-3.1c — inRankFile → LTR (MAP) → LTR (NDCG@10)

**Pipeline:**
1. Pre-existing ranked list (inRankFile)
2. LTR stage 1 (RankLib model 4, MAP metric, rerankDepth=500, BM25: k₁=2.0 b=0.3, QL: mu=2500, featureDisable=20)
3. LTR stage 2 (RankLib model 4, NDCG@10 metric, rerankDepth=100, BM25: k₁=1.2 b=0.75, QL: mu=2500, featureDisable=20)

**Intuition:** A pure LTR cascade with different optimization targets at each stage. The first stage uses MAP as the training metric (broader recall-precision tradeoff) with aggressive BM25 parameters (high k₁=2.0, low b=0.3 to favor term frequency and penalize length less), pruning from ~all docs down to 500. The second stage optimizes directly for NDCG@10 with standard BM25 tuning, targeting top-list quality. The hypothesis was that optimizing for MAP first preserves recall while a second pass explicitly targets graded relevance ordering at the top of the list.

---

## Exp-3.1d — inRankFile → PRF/RM3 → BM25 → BERT (L-6)

**Pipeline:**
1. Pre-existing ranked list (inRankFile)
2. RM3 pseudo-relevance feedback (numDocs=25, numTerms=3, expansionField=body)
3. BM25 re-retrieval (k₁=1.2, b=0.75, top 500)
4. MiniLM-L-6-v2 maxp rerank (rerankDepth=500, psgLen=150, psgStride=125, psgCnt=6)

**Intuition:** Tested whether pseudo-relevance feedback could improve the pipeline by expanding queries with terms from the top-25 initial results before re-running BM25. The hypothesis was that RM3 would surface additional relevant documents missed by the original query, and BERT would then rerank the expanded result set effectively. In practice, this was the worst-performing configuration, suggesting that RM3 expansion caused significant query drift — adding noisy terms that pulled BM25 retrieval away from true relevance.

**Results:** MAP=0.2125, NDCG@20=0.4531, P@5=0.1500 — **worst performer**

---

## Exp-3.1e — inRankFile → BERT (L-6, firstp) → BERT (L-12, maxp)

**Pipeline:**
1. Pre-existing ranked list (inRankFile)
2. MiniLM-L-6-v2 **firstp** rerank (rerankDepth=800, psgLen=150, psgStride=125, psgCnt=6)
3. MiniLM-L-12-v2 **maxp** rerank (rerankDepth=100, psgLen=150, psgStride=125, psgCnt=6)

**Intuition:** Tests whether different passage aggregation strategies can complement each other in a cascade. `firstp` (score based only on the first passage) is fast and rewards documents where the query topic appears prominently near the top — a useful proxy for on-topic, well-structured documents. Then `maxp` (best-passage score) is applied to the top 100 to get the most accurate relevance signal for the final ranking. The large rerankDepth=800 for the first stage ensures minimal recall loss. However, results were poor, possibly because `firstp` is too coarse a signal and loses too many relevant documents during the broad pruning pass.

**Results:** MAP=0.2397, NDCG@20=0.4608, P@5=0.2667

---

## Exp-3.1f — RankedBoolean → BERT (L-6) → BERT (L-12)

**Pipeline:**
1. RankedBoolean retrieval (top 1000)
2. MiniLM-L-6-v2 maxp rerank (rerankDepth=300, psgLen=150, psgStride=125, psgCnt=3)
3. MiniLM-L-12-v2 maxp rerank (rerankDepth=100, psgLen=150, psgStride=125, psgCnt=3)

**Intuition:** Tests whether BERT reranking can compensate for a weaker initial retrieval model. RankedBoolean uses boolean matching with IDF-style scoring rather than BM25's full TF-IDF weighting, producing a cruder initial list. The hypothesis was that BERT's strong semantic understanding might be sufficient to recover relevant documents regardless of initial ranking quality — i.e., testing the robustness of BERT reranking to the choice of base retrieval model. Results were surprisingly competitive with Exp-3.1a (which used BM25 as the base), supporting the hypothesis that BERT is fairly robust to initial retrieval quality.

**Results:** MAP=0.3932, NDCG@20=0.6267, P@5=0.5000

---

## Summary

| Experiment | Pipeline | Key Hypothesis | MAP | NDCG@20 | P@5 |
|-----------|---------|---------------|-----|---------|-----|
| 3.1a | BM25 → BERT-L6 → BERT-L12 | Cascaded BERT reranking baseline | 0.3815 | 0.6256 | 0.4833 |
| 3.1b | inRank → LTR → BERT-L12 | LTR + BERT signals are complementary | **0.3885** | **0.6361** | **0.5000** |
| 3.1c | inRank → LTR(MAP) → LTR(NDCG) | Two-stage LTR with different optimization targets | — | — | — |
| 3.1d | inRank → RM3 → BM25 → BERT | PRF query expansion improves downstream reranking | 0.2125 | 0.4531 | 0.1500 |
| 3.1e | inRank → BERT(firstp) → BERT(maxp) | Complementary aggregation strategies in BERT cascade | 0.2397 | 0.4608 | 0.2667 |
| 3.1f | RankedBoolean → BERT-L6 → BERT-L12 | BERT is robust to a weaker initial ranker | 0.3932 | 0.6267 | 0.5000 |
