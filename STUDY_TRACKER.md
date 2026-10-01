# CMPE 256 Midterm — Study Tracker ✅
**Target: Oct 8, 2026** — tick each box (`[ ]` → `[x]`) as you finish.

> Suggested plan: ~1 session deck per day, then 2 days of practice problems + review.

---

## 📅 Day-by-day plan

### Day 1 — Session 1: Intro & Data Mining (08/20) ⭐⭐⭐
- [ ] Read `CMPE256_Recommender Systems_Session_1_20260820.pdf`
- [ ] Define data mining & KDD (know the "non-trivial, implicit, unknown, useful" phrasing)
- [ ] Memorize the **4 areas**: classification, segmentation, association, anomaly detection
- [ ] Memorize the **7 KDD process steps** in order
- [ ] Supervised vs unsupervised vs reinforcement; lazy learning
- [ ] Structured/semi/unstructured data + the "V"s (volume, velocity, variety, veracity)

### Day 2 — Recommendation Systems & Content-Based (08/27) ⭐⭐⭐
- [ ] Read `CMPE256_RecommenderSystems_Session_RecommendationSystems_20260827.pdf`
- [ ] Content-based vs collaborative filtering (define & contrast)
- [ ] Utility matrix: users/items, sparse, "predict the blanks"
- [ ] Long-tail phenomenon (physical vs online)
- [ ] Item profiles (movie features: actors, director, year, genre)
- [ ] NLP/NLTK preprocessing pipeline (tokenize → lowercase → punctuation → stopwords → stem/lemmatize)
- [ ] Stemming vs lemmatization; entity extraction; POS tagging
- [ ] Vector Space Model

### Day 3 — TF-IDF & Cosine Similarity (08/27, part 2) ⭐⭐⭐
- [ ] Understand TF, IDF, and TF-IDF = TF × IDF
- [ ] Write the cosine similarity formula from memory
- [ ] Know value meanings: +1 / 0 / −1
- [ ] Work a cosine similarity example by hand (two vectors)
- [ ] Work a TF-IDF example by hand
- [ ] Skim demo code: `InClassDemoCode/D_cosineSimilarity*.py`

### Day 4 — Clustering (09/03) ⭐⭐⭐
- [ ] Read `CMPE256_DataMining_Session_3_Clustering_20260903.pdf`
- [ ] Clustering = unsupervised; small intra-cluster, large inter-cluster distance
- [ ] Distance metrics: Euclidean (L2), Manhattan (L1), Minkowski (Lp)
- [ ] Work x1=(1,2), x2=(3,5) → Euclidean & Manhattan
- [ ] Binary variables: q/r/s/t table; symmetric vs asymmetric
- [ ] Jaccard coefficient & asymmetric dissimilarity formulas
- [ ] Hierarchical: agglomerative vs divisive; dendrogram
- [ ] Linkage: single / complete / average / Ward
- [ ] k-means algorithm (write pseudo-code) + its limitations
- [ ] Alternatives when k unknown: GMM, DBSCAN, mean-shift

### Day 5 — Information Retrieval & Web Search (09/10) ⭐⭐
- [ ] Read `CMPE256_LargeScaleAnalytics_Session_InformationRetrieval_20260910.pdf`
- [ ] IR vs data retrieval
- [ ] Crawlers/spiders; depth-first vs breadth-first (pros/cons, which is standard)
- [ ] Robots Exclusion Protocol; canonical URLs
- [ ] HTML structure basics; `<head>`/meta/DOCTYPE
- [ ] SEO: indexing, back links, "can't pay for organic rank"
- [ ] (If lectured before exam) link analysis / ranking intuition

### Day 6 — Practice & formula drill
- [ ] Redo all 4 numeric problem types without notes
- [ ] Rewrite the formula sheet from memory (section 5 of study guide)
- [ ] Do the 10 likely-question types in the study guide (section 6)
- [ ] Re-skim every deck's section headers for anything missed

### Day 7 — Final review (day before)
- [ ] Flash-review definitions (fill-in-the-blank practice)
- [ ] Re-derive each formula once
- [ ] Light review only — rest well 💤

---

## 🧮 Must-be-able-to-compute by hand
- [ ] Cosine similarity between two vectors
- [ ] TF-IDF for a (document, term)
- [ ] Euclidean & Manhattan distance
- [ ] One iteration of k-means (assign → recompute centroids)
- [ ] Jaccard / asymmetric binary dissimilarity from q,r,s,t

## 🧠 Must-be-able-to-define/contrast
- [ ] Content-based vs collaborative filtering
- [ ] Utility matrix + long tail
- [ ] KDD 7 steps (ordered)
- [ ] NLP preprocessing pipeline (ordered)
- [ ] Supervised vs unsupervised vs reinforcement
- [ ] Depth-first vs breadth-first crawling
- [ ] Agglomerative vs divisive; single/complete/average linkage
- [ ] IR vs data retrieval; stemming vs lemmatization; TF vs IDF

## ✍️ Whiteboard / pseudo-algorithm
- [ ] k-means algorithm
- [ ] Content-based recommender high-level flow

---

## ⏭️ Out of scope for THIS midterm (skip for now)
These are on the roadmap for **after** 10/08 — don't spend time here:
- [ ] ~~RAG / KAG~~
- [ ] ~~LLMs / Generative AI~~
- [ ] ~~Agents / A2A / multi-agent orchestration~~
- [ ] ~~TimeGPT / time series~~
- [ ] ~~Audio/Video modalities~~
- [ ] ~~Edge/streaming recommender architecture~~
- [ ] ~~Azure OpenAI / LLMOps PDFs (reference only)~~
