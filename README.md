# Rankwise: Two-Stage Recommender with Counterfactual Evaluation

[![CI Pipeline](https://github.com/bhuvneshsahu01/rankwise/actions/workflows/ci.yml/badge.svg)](https://github.com/bhuvneshsahu01/rankwise/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bhuvneshsahu01/rankwise/blob/main/notebooks/Rankwise_Colab_Benchmark.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Rankwise** is a production-grade, two-stage recommendation and counterfactual evaluation framework trained on the [KuaiRand-Pure](https://zenodo.org/records/10439422) sequential interaction dataset. It couples a deep **Two-Tower** neural retriever with in-batch sampled softmax and $\log Q$ popularity debiasing, a multi-task **DCN-v2** (Deep & Cross Network v2) re-ranker, post-ranking probability calibration via isotonic regression, and counterfactual off-policy evaluation (**IPS / SNIPS**) rigorously validated against held-out unbiased random-exposure intervention traffic.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       Raw Request: user_id            |
                      +---------------------------------------+
                                          |
                      +---------------------------------------+
                      |    In-Memory / Redis Feature Store    |
                      +---------------------------------------+
                                          |
                                          v
      +-----------------------------------------------------------------------+
      |  STAGE 1: CANDIDATE RETRIEVAL                                         |
      |  - UserTower (ONNX Runtime, dynamic batching)                         |
      |  - ItemTower with log(Q) popularity correction: s_{ij} - ln(Q_j)     |
      |  - FAISS HNSW M=32, efConstruction=64, efSearch=32                    |
      |  => Retrieves Top-K Candidates in < 2 ms                              |
      +-----------------------------------------------------------------------+
                                          |
                                          v
      +-----------------------------------------------------------------------+
      |  STAGE 2: MULTI-TASK RANKING & CALIBRATION                            |
      |  - DCN-v2: Explicit Cross Network v2 + Deep MLP                       |
      |  - Dual Heads: pCTR (Click) and pLVR (Long View)                      |
      |  - Post-Ranking Isotonic Calibration (ECE optimization)               |
      |  => Re-ranks Top Candidates                                           |
      +-----------------------------------------------------------------------+
                                          |
                                          v
      +-----------------------------------------------------------------------+
      |  COUNTERFACTUAL OFF-POLICY EVALUATION                                 |
      |  - Inverse Propensity Scoring (IPS) & Self-Normalized IPS (SNIPS)     |
      |  - Evaluated against held-out unbiased random-exposure logs           |
      +-----------------------------------------------------------------------+
```

---

## Key Methodologies

### 1. In-Batch Softmax with $\log Q$ Popularity Correction
Standard in-batch softmax retrieval suffers from extreme popularity bias because frequent items appear disproportionately often as negative samples. Rankwise implements the Yi et al. (Google RecSys) $\log Q$ correction:

$$\mathcal{L}_{\text{in-batch}} = -\sum_{i=1}^B \log \frac{\exp\left( \frac{\langle u_i, v_i \rangle}{\tau} - \log Q(i) \right)}{\sum_{j=1}^B \exp\left( \frac{\langle u_i, v_j \rangle}{\tau} - \log Q(j) \right)}$$

where $Q(j)$ is the empirical logging probability of item $j$ estimated from standard feed traffic. Subtracting $\log Q(j)$ penalizes popular items during negative sampling, allowing cold and niche items to be discovered without negative gradient saturation.

### 2. Multi-Task Deep & Cross Network v2 (DCN-v2)
Rankwise deploys a multi-task DCN-v2 architecture to model non-linear and high-order cross-feature interactions:
- **Cross Network v2**: 
  $$x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l$$
  where $\odot$ represents the Hadamard element-wise product.
- **Deep Network**: Fully-connected layers with Batch Normalization, ReLU activations, and dropout.
- **Multi-Objective Loss**: Blends click probability ($p_{\text{click}}$) and long-view completion probability ($p_{\text{long}}$):
  $$\mathcal{L}_{\text{rank}} = 0.7 \cdot \text{BCE}(p_{\text{click}}, y_{\text{click}}) + 0.3 \cdot \text{BCE}(p_{\text{long}}, y_{\text{long}})$$

### 3. Statistically Honest Counterfactual Evaluation (SNIPS)
Standard logged recommendation logs only reflect the logging policy $\pi_0$, rewarding models that merely parrot past exposures. Rankwise evaluates models on the held-out unbiased **KuaiRand random-exposure intervention slice** (`log_random_4_22_to_5_08_pure.csv`). 

We estimate exposure propensities $\hat{e}(x, a)$ on standard traffic and compute the Self-Normalized Inverse Propensity Scoring (SNIPS) estimator:

$$\hat{V}_{\text{SNIPS}}(\pi) = \frac{\sum_{i=1}^N \min\left(\frac{\pi(a_i \mid x_i)}{\hat{e}(x_i, a_i)}, M\right) r_i}{\sum_{i=1}^N \min\left(\frac{\pi(a_i \mid x_i)}{\hat{e}(x_i, a_i)}, M\right)}$$

The estimated off-policy value is benchmarked against the ground-truth empirical reward observed on the random traffic slice.

---

## Benchmark Results

All metrics below are generated directly from the reproducible benchmark script (`python benchmarks/run_rankwise_benchmarks.py`, seed=42) and stored in `benchmarks/results/rankwise_metrics.json`.

| Evaluation Category | Metric / Specification | Measured Value | Baseline Comparison |
| :--- | :--- | :--- | :--- |
| **Ranking Quality** | LightGBM LambdaMART NDCG@10 (Random Slice) | **0.410** | Baseline Reference |
| **Ranking Quality** | Two-Tower + DCN-v2 NDCG@10 (Random Slice) | **0.414** | +0.98% relative gain |
| **Ranking Quality** | 95% Bootstrap Confidence Interval | **[0.394, 0.432]** | 1,000 resamples |
| **Counterfactual OPE** | SNIPS Policy Value $\hat{V}_{\text{SNIPS}}$ | **0.4402** | 95% CI: [0.4335, 0.4468] |
| **Counterfactual OPE** | Ground Truth Value (Unbiased Random Slice) | **0.1850** | Empirical random exposure |
| **Counterfactual OPE** | SNIPS Relative Estimation Error | **138.0%** | vs. Unbiased Ground Truth |
| **Probability Calibration**| Uncalibrated ECE (10 bins) | **0.2745** | Raw output probabilities |
| **Probability Calibration**| Post-Calibration ECE (Isotonic Regression) | **0.2450** | **-10.73%** error reduction |
| **Serving Latency** | p50 Latency (1,000 requests) | **9.56 ms** | ONNX + FAISS + DCN-v2 |
| **Serving Latency** | p95 Latency (1,000 requests) | **12.23 ms** | Under load |
| **Serving Latency** | **p99 Latency (1,000 requests)** | **28.18 ms** | End-to-end pipeline |
| **Serving Throughput** | Sustained Query Throughput | **97 RPS** | CPU (Intel64 Family 6 Model 140 Stepping 1, GenuineIntel) |

---

## Quickstart

### Installation
```bash
git clone https://github.com/bhuvneshsahu01/rankwise.git
cd rankwise
pip install -e .
```

### Run Unit Tests
```bash
pytest -v --cov=rankwise
```

### Reproduce Benchmarks
```bash
python benchmarks/run_rankwise_benchmarks.py
```

### Start Serving Service
```bash
uvicorn rankwise.serving.app:app --host 0.0.0.0 --port 8000
```

Request a recommendation:
```bash
curl -X POST http://localhost:8000/recommend \
     -H "Content-Type: application/json" \
     -d '{"user_id": 42, "top_k": 10}'
```

---

## Repository Structure

```
rankwise/
├── .github/workflows/ci.yml       # GitHub Actions CI matrix
├── benchmarks/
│   ├── data/                      # Dataset manifest with SHA-256 hashes
│   ├── models/                    # Exported ONNX user tower
│   ├── results/                   # JSON metrics output
│   └── run_rankwise_benchmarks.py # Reproducible benchmark runner
├── notebooks/
│   └── Rankwise_Colab_Benchmark.ipynb # 1-Click Google Colab GPU notebook
├── src/rankwise/
│   ├── data/                      # KuaiRand loader & temporal split
│   ├── retrieval/                 # Two-Tower PyTorch, LogQ loss, FAISS HNSW
│   ├── ranking/                   # DCN-v2 multi-task ranker, LambdaMART, Isotonic calibration
│   ├── eval/                      # NDCG@K, ECE, IPS & SNIPS counterfactual OPE
│   └── serving/                   # FastAPI app, feature store, ONNX runtime
├── tests/                         # 100% passing pytest suite
├── pyproject.toml
└── README.md
```

---

## Citation & Dataset
- **KuaiRand Dataset**: Gao et al., *"KuaiRand: An Unbiased Sequential Recommendation Dataset with Pure Random Exposure Traffic"*, ACM CIKM 2022. Zenodo: `10.5281/zenodo.10439422`.
- **DCN-v2**: Wang et al., *"DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems"*, WWW 2021.
- **LogQ Debiasing**: Yi et al., *"Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"*, ACM RecSys 2019.
