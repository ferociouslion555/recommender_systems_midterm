# CMPE 256 — Recommender Systems — Midterm Study Guide
**Exam date:** Wednesday, Oct 8, 2026 · **Weight:** 15% of final grade
**Instructor:** Prof. Chandrasekar Vuppalapati, SJSU

---

## 0. Which slides to study (scope)

The class roadmap shows the midterm (10/08) comes right after these sessions. **Everything from 08/20 → 09/24 is fair game.** Prioritize in this order:

| Priority | Slide deck (file) | Topic | Dates |
|----------|-------------------|-------|-------|
| ⭐⭐⭐ HIGH | `CMPE256_RecommenderSystems_Session_RecommendationSystems_20260827.pdf` | Recommendation systems, content-based, TF-IDF, cosine similarity, NLP/NLTK | 08/27 |
| ⭐⭐⭐ HIGH | `CMPE256_Recommender Systems_Session_1_20260820.pdf` | Intro, data mining, KDD process, ML basics | 08/20 |
| ⭐⭐⭐ HIGH | `CMPE256_DataMining_Session_3_Clustering_20260903.pdf` | Clustering, distance metrics, k-means, hierarchical | 09/03 |
| ⭐⭐ MED | `CMPE256_LargeScaleAnalytics_Session_InformationRetrieval_20260910.pdf` | Information retrieval, crawlers, HTML/SEO, vector space | 09/10 |
| ⭐ SUPPORT | `InClassDemoCode/*.py` | Cosine similarity demos (spaCy, TF-IDF, CountVectorizer) | — |

**Likely NOT on this midterm** (these are post-midterm / 10/15+ on the roadmap): RAG, KAG, LLMs, agents/A2A, TimeGPT, audio/video modalities, edge/streaming. The Azure/LLMOps PDFs in the repo are reference material, not core midterm content — skim only if you have time.

> **Exam format hints from the slides:** whiteboard solutions, write high-level pseudo-algorithms, multiple-choice, fill-in-the-blanks. There are **graded in-class examples** on asymmetric binary dissimilarity, k-means step-by-step, and TF-IDF calculation — expect numeric problems like these.

---

## 1. Introduction & Data Mining (08/20)

### What is Data Mining?
- **Definition:** discovering meaningful new correlations, patterns, and trends by sifting through large amounts of data using pattern recognition + statistical/mathematical techniques.
- Also called **KDD — Knowledge Discovery in Databases**: extraction of *interesting (non-trivial, implicit, previously unknown, potentially useful)* information/patterns from large databases.
- Driver: **data explosion** — "We are drowning in data, but starving for knowledge." Solution = data warehousing + data mining + OLAP.
- Data mining is an **interactive and iterative** process (not magic / not fully automatic).

### Four basic areas of data mining exploration
1. **Classification** — assign a record to a known group/value.
2. **Segmentation (clustering)** — create groups/segments from data.
3. **Association** — find things that occur together or in sequence (bread → milk).
4. **Anomaly detection** — find items that don't fit the pattern (fraud detection).

- **Directed (supervised)** vs **undirected (unsupervised)** data mining.

### KDD process steps (know the order!)
1. Data **cleaning** (remove noise, inconsistent data)
2. Data **integration** (combine multiple sources)
3. Data **selection** (retrieve relevant data)
4. Data **transformation** (consolidate/aggregate into mining-ready form)
5. Data **mining** (apply intelligent methods to extract patterns)
6. **Pattern evaluation** (identify truly interesting patterns)
7. **Knowledge presentation** (visualize/represent results)

### ML basics (vocabulary)
- ML = subset of AI; learns & improves from data **without being explicitly programmed**.
- **Supervised** (labeled), **Unsupervised** (finds structure on its own), **Reinforcement** (learns from good/bad evaluations — games, driving).
- **Lazy learning** — delays processing until test data arrives (case-based; e.g., kNN); less time training, more time predicting.

### Big data context
- Structured / Semi-structured / Unstructured data. Sources: transactional/application, machine, social, enterprise content.
- "Vs" to remember: Volume, Velocity, Variety, **Veracity**, (Throughput, Ingestion).

### Course grading (good fill-in-the-blank material)
In-class 5% · Homework 5% · Pop quizzes 10% · Team research paper/hackathon 15% · Team project 20% · **Midterm 15%** · Final 30%.

---

## 2. Recommendation Systems & Content-Based (08/27) ⭐ most important

### Core concept
- A **recommendation system** predicts user responses to options (predicting user preferences for items).
- **Two broad classes:**
  - **Content-based** — examine *properties of the items*; recommend items similar in their properties (e.g., watched many cowboy movies → recommend cowboy movies).
  - **Collaborative filtering** — recommend based on *similarity between users and/or items*; recommend what similar users preferred.

### The Utility Matrix
- Two classes of entities: **users** and **items**.
- Utility matrix = for each (user, item) pair, a value representing degree of preference (e.g., 1–5 stars).
- The matrix is **sparse** (most entries blank).
- **Goal of a recommender = predict the blanks in the utility matrix.**

### The Long Tail
- **Physical** stores: limited shelf space → can show only a small fraction of items (scarcity of resources).
- **Online**: can offer everything, but users can't browse it all → the **long-tail phenomenon forces online platforms to recommend** items to individual users.

### Applications
- Product recommendation (Amazon), movie recommendation (Netflix), blogs, YouTube videos.

### Content-based: Item Profiles
- Build a **profile** (record of important characteristics) for each item.
- Movie features example: **set of actors, director, year, genre**.
- For documents: features aren't readily available → identify **words that characterize the topic** of the document.
  - Compare document word-sets using **Jaccard distance** or **cosine distance** (treat as vectors).

### NLP / NLTK pipeline (text preprocessing — know the steps)
1. Load raw text
2. Tokenize (split into tokens/words) — `word_tokenize`, `sent_tokenize`
3. Convert to lowercase
4. Remove punctuation
5. Filter non-alphabetic tokens
6. Remove **stop words**
7. **Stemming** (Porter Stemmer — chops to word stem: drug/drugged/drugs → drug)
8. **Lemmatization** — map to dictionary form ("running","ran" → "run")

- **NLP tasks:** sentiment analysis, topic detection, language detection, key-phrase extraction, document categorization, spam/sensitive labeling.
- **NLP techniques:** tokenizer, stemming/lemmatization, **entity extraction**, part-of-speech (POS) detection, sentence boundary detection.
- Tools: NLTK modules (`nltk.corpus`, `nltk.tokenize`, `nltk.stem`, `nltk.collocations`), spaCy (`nlp(text)`, `doc.ents`).
- POS tags worth recognizing: NN noun, VB verb, RB adverb, etc.

### Vector Space Model (VSM)
- Represent **document and query both as vectors** in a high-dimensional space (one dimension per keyword/term).
- Use a similarity measure to compare query vector vs document vector → rank documents.
- First identify **keywords**; group words sharing a **stem** (drug, drugged, drugs → drug).

### TF-IDF (★ expect a calculation problem)
- **TF (term frequency):** number of occurrences of term *t* in document *d*, `freq(d,t)`. 0 if term absent. Often normalized (Cornell SMART formula).
- **IDF (inverse document frequency):** scaling factor / importance of a term. A term appearing in **many** documents has **less** discriminative power → its importance is **scaled down**.
- **TF-IDF = TF × IDF** — combines both; high for terms frequent in a doc but rare across the corpus.

### Cosine Similarity (★ expect a calculation problem)
- Measures similarity by the **angle** between vectors, **not magnitude**.
- Formula: **cos(θ) = (A · B) / (‖A‖ ‖B‖)** = dot product ÷ (product of vector lengths/magnitudes).
- Values: **+1 = same direction (identical), 0 = 90° (unrelated), −1 = opposite.**
- Worked example idea: movies as vectors of (actors + rating); dot product of binary actor vectors + rating term, divided by the two vector lengths.
- **User profiles:** blanks can be treated as 0 (questionable since 0 ≠ "no opinion"). Compare users by cosine of angle between their rating vectors.

### Classification: Decision Trees
- A **decision tree** = nodes arranged as a binary tree; leaves render decisions; internal nodes test a predicate on item features.

### Bag of Words / Document Vector
- Represent text as word-count vectors (Bag of Words). Convert text → sequence → fixed-length document vector for classification.
- Demo tools: `CountVectorizer`, `TfidfVectorizer`, `cosine_similarity` (sklearn).

---

## 3. Clustering (09/03) ⭐

### What is clustering?
- **Grouping objects into classes of similar objects** by a distance measure.
- Goal: **small distance within a cluster, large distance between clusters.**
- Also called **data segmentation**; can be used for **outlier detection**.
- Clustering = **unsupervised learning** → learning by **observation**, not by examples (no predefined classes/labels).
- Classic algorithms: **k-means, k-medoids**.

### Data representations
- **Data matrix** (object-by-variable): n objects × p variables (n×p).
- **Dissimilarity matrix** (object-by-object): n×n, where `d(i,j)` = dissimilarity (≈0 when similar, larger when more different).

### Distance metrics (★ know formulas)
- **Euclidean (L2):** `sqrt( Σ (x_i − y_i)² )`
- **Manhattan / city-block (L1):** `Σ |x_i − y_i|`
- **Minkowski (Lp):** `( Σ |x_i − y_i|^p )^(1/p)` — generalizes both: p=1 → Manhattan, p=2 → Euclidean.
- **Weighted** versions multiply each dimension by a weight.
- Worked example: x1=(1,2), x2=(3,5) → Euclidean = √(4+9)=√13 ≈ 3.61; Manhattan = 2+3 = 5.

### Standardization
- Interval-scaled variables = continuous (weight, height, temperature, lat/long).
- **Measurement units affect clustering** → standardize so all variables get equal weight.

### Binary variables (★ graded in-class example)
Contingency table for objects i, j over p binary variables:
- **q** = # vars = 1 for both i and j
- **r** = # vars = 1 for i, 0 for j
- **s** = # vars = 0 for i, 1 for j
- **t** = # vars = 0 for both
- p = q + r + s + t

- **Symmetric** binary var: both states equally valuable (e.g., gender). Symmetric dissimilarity = `(r + s) / (q + r + s + t)`.
- **Asymmetric** binary var: states unequal; rarest/important outcome coded 1 (e.g., HIV positive). Negative matches (t) ignored.
  - **Asymmetric dissimilarity** = `(r + s) / (q + r + s)`
  - **Jaccard coefficient (similarity)** = `q / (q + r + s)` = 1 − asymmetric dissimilarity.
- Worked example from slides: d(Anitha,Tien) = (2+2)/(1+2+2) = 4/5 = 0.8; d(Anitha,Tami) = (1+2)/(2+1+2) = 3/5 = 0.6.

### Hierarchical clustering
- **Agglomerative (bottom-up):** start with each point as its own cluster, successively **merge** closest clusters. (Most common.)
- **Divisive (top-down):** start with all points in one cluster, successively **split**.
- Visualized by a **dendrogram** (tree showing merge order / relations).
- Does **not** require k up front, but needs a **termination condition**.
- **Linkage methods** (how to measure cluster-to-cluster distance):
  - **Single link** — distance between the *closest* pair of points.
  - **Complete link** — *maximum* distance between pairs.
  - **Average link** — *average* of all pairwise distances.
  - **Ward** — minimizes within-cluster sum of squared differences (variance-minimizing; similar objective to k-means).
- Radius vs diameter of a cluster: radius measured from centroid; diameter = greatest distance between two points (diameter ≈ but not exactly 2×radius, since centroid may not lie on the line between extreme points).

### k-means clustering (★ step-by-step problem likely)
- Searches for a **pre-determined number k** of clusters in an unlabeled multidimensional dataset.
- **Algorithm:**
  1. Choose k and pick initial cluster means/centroids (e.g., the two points furthest apart).
  2. Assign each point to the **nearest centroid** (by Euclidean distance).
  3. **Recompute** each centroid as the mean of its assigned points.
  4. **Repeat until converged** (assignments stop changing).
- **Limitations:**
  - **k must be chosen beforehand** — can't learn k from data.
  - Limited to **linear/convex cluster boundaries**.
  - Sensitive to initialization.
- Alternatives when k is unknown or boundaries nonlinear: **Gaussian Mixture Models (GMM), DBSCAN (density-based), mean-shift, affinity propagation.**
- IoT/edge example in slides: cluster sensors by **RSSI (signal strength)** and distance.

---

## 4. Information Retrieval & Web Search (09/10) ⭐⭐ (medium)

### IR fundamentals
- **IR** = representing, storing, organizing, and offering access to information items; helping users find info matching their **information needs**.
- **IR vs Data Retrieval:** IR works on **unstructured/free-form** text & multimedia; data retrieval finds **precise** data in **structured** databases.
- Two ways to search: **search engines** vs **browsing directories** (categories, e.g., Yahoo Directories).

### Search engine architecture / crawlers
- **Web crawlers (spiders/robots):** collect webpages to build the text collection. Text extracted from HTML; headings/bold can get **higher weight**.
- Start from **root URLs**, follow links recursively.
- **Depth-first search:** go down one branch to a leaf before backtracking. Uses **less memory / smaller fringe**; **not complete, not optimal**.
- **Breadth-first search:** explore level by level, uniformly outward. **Complete and optimal** but **memory-intensive** (exponential in depth). **Standard spidering method.**
- Links put into **canonical form** (strip trailing slash, remove internal references, default to index.html).
- **Robots Exclusion Protocol / robots meta tag** — tells crawlers not to index certain areas.

### HTML essentials
- HTML describes the **structure** of web pages via **markup/elements**.
- `<!DOCTYPE html>` must come first (case-insensitive). `<head>` holds metadata (title, styles, scripts) — not displayed.
- Meta data names: application-name, author, description, generator, keywords.

### SEO (Search Engine Optimization)
- Search engines scan & **index** site code/content to decide when/where a site appears on results.
- Page content should be inviting, comprehensive, keyword-rich (within reason).
- **Back links** (external sites linking to you) heavily influence importance/ranking (esp. Google).
- SEO takes constant tweaking; you **cannot pay to guarantee** organic ranking.

### Link analysis (preview — on roadmap for 09/17–09/24)
- Ranking by link structure (PageRank-style), rank analysis, association rules. If covered in a lecture before 10/08, review the ranking intuition: importance of a page depends on importance of pages linking to it.

---

## 5. Quick formula sheet (memorize)

| Concept | Formula |
|---------|---------|
| Cosine similarity | cos(θ) = (A·B) / (‖A‖·‖B‖) |
| Euclidean (L2) | √( Σ (xᵢ − yᵢ)² ) |
| Manhattan (L1) | Σ \|xᵢ − yᵢ\| |
| Minkowski (Lp) | ( Σ \|xᵢ − yᵢ\|^p )^(1/p) |
| TF-IDF | TF(d,t) × IDF(t) |
| Jaccard coefficient (sim) | q / (q + r + s) |
| Asymmetric binary dissimilarity | (r + s) / (q + r + s) |
| Symmetric binary dissimilarity | (r + s) / (q + r + s + t) |

---

## 6. Likely exam question types (based on slide cues)
1. **Numeric:** compute cosine similarity between two document/user vectors.
2. **Numeric:** compute TF-IDF for a given (doc, term).
3. **Numeric:** compute Euclidean/Manhattan distance; run one iteration of k-means.
4. **Numeric:** asymmetric binary dissimilarity / Jaccard from a contingency table.
5. **Conceptual:** content-based vs collaborative filtering — define & contrast.
6. **Conceptual:** explain the utility matrix and the long-tail phenomenon.
7. **Ordering/fill-in:** steps of the KDD process; the NLP preprocessing pipeline.
8. **Compare:** depth-first vs breadth-first crawling; agglomerative vs divisive; single/complete/average linkage.
9. **Definitions:** IR vs data retrieval, stemming vs lemmatization, supervised vs unsupervised, TF vs IDF.
10. **Pseudo-algorithm (whiteboard):** write the k-means algorithm.
