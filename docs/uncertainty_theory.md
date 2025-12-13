# Uncertainty Quantification Theory for RAG

## Overview

This document provides the theoretical foundation for uncertainty quantification in Retrieval-Augmented Generation (RAG) systems. We present a novel framework combining Bayesian inference with conformal prediction to provide calibrated confidence intervals.

## 1. The Problem: Uncertainty in RAG

RAG systems introduce a unique challenge: **uncertainty propagates through two stages**:

1. **Retrieval Uncertainty**: Did we retrieve the right documents?
2. **Generation Uncertainty**: Does the LLM answer correctly given the context?

Traditional LLM uncertainty methods only address (2). Our framework addresses both.

### Formal Setup

Let:
- $q$ = query
- $\mathcal{D}$ = document corpus
- $r: q \rightarrow \mathcal{D}^k$ = retrieval function (returns top-k documents)
- $g: (q, d_{1:k}) \rightarrow a$ = generation function
- $a^*$ = ground truth answer

The RAG pipeline is:
$$a = g(q, r(q))$$

We want to quantify:
$$P(a = a^* | q, \mathcal{D})$$

## 2. Retrieval Uncertainty

### 2.1 Hybrid Retrieval with Learned Gating

Given two retrieval functions:
- $r_{BM25}(q)$: Sparse (lexical) retrieval
- $r_{dense}(q)$: Dense (semantic) retrieval

We learn a gating function $w_\theta: (s_{BM25}, s_{dense}) \rightarrow [0, 1]$ that outputs per-passage weights.

The hybrid score for passage $p$ is:
$$s_{hybrid}(p) = w_\theta(p) \cdot s_{dense}(p) + (1 - w_\theta(p)) \cdot s_{BM25}(p)$$

**Training Objective**: Maximize retrieval quality via differentiable ApproxNDCG:

$$\mathcal{L}(\theta) = -\mathbb{E}_q\left[\text{ApproxNDCG}(s_{hybrid}, \text{relevance})\right]$$

### 2.2 Retrieval Uncertainty Estimation

We estimate retrieval uncertainty through:

1. **Score Variance**: High variance in top-k scores indicates uncertainty
2. **Score Gap**: Small gap between top-1 and top-k suggests ambiguity
3. **Router Confidence**: Extreme weights (0 or 1) indicate clear preference

$$U_{retrieval} = \sigma(s_{top-k}) + \lambda \cdot (1 - |s_1 - s_k|)$$

## 3. Generation Uncertainty

### 3.1 Monte Carlo Dropout

Following [Gal & Ghahramani, 2016], we treat dropout as approximate Bayesian inference.

For an LLM with dropout layers, we sample $T$ predictions:
$$\{a_t\}_{t=1}^T = \{g(q, d_{1:k}; \omega_t)\}_{t=1}^T$$

where $\omega_t$ are stochastic dropout masks.

**Uncertainty Estimation**:
$$U_{MC} = \text{Var}(\phi(a_1), ..., \phi(a_T))$$

where $\phi(\cdot)$ is a sentence embedding function.

**Practical Implementation**: Since Ollama doesn't support native dropout:
- We use temperature variation as a proxy
- High temperature ($T > 0.8$) samples from broader distribution
- Top-p variation adds additional stochasticity

### 3.2 Embedding Space Variance

Let $e_t = \phi(a_t)$ be the embedding of answer $t$.

**Centroid**:
$$\bar{e} = \frac{1}{T}\sum_{t=1}^T e_t$$

**Variance**:
$$U_{embedding} = \frac{1}{T}\sum_{t=1}^T \|e_t - \bar{e}\|_2$$

**Consensus Answer**: 
$$a^* = \arg\min_{a_t} \|e_t - \bar{e}\|_2$$

## 4. Conformal Prediction

### 4.1 Framework

Conformal prediction [Vovk et al., 2005] provides distribution-free coverage guarantees.

**Key Property**: Given miscoverage level $\alpha$ and calibration set of size $n$:
$$P(a^* \in C(q)) \geq 1 - \alpha$$

where $C(q)$ is the prediction set.

### 4.2 Nonconformity Score

We define the nonconformity score as:
$$s(q, a) = 1 - \text{ROUGE-L}(a, a^*)$$

Higher scores indicate the prediction is "less conforming" to ground truth.

### 4.3 Calibration

For calibration set $\{(q_i, a^*_i)\}_{i=1}^n$:

1. Compute predictions: $\hat{a}_i = g(q_i, r(q_i))$
2. Compute scores: $s_i = 1 - \text{ROUGE-L}(\hat{a}_i, a^*_i)$
3. Store sorted scores: $s_{(1)} \leq s_{(2)} \leq ... \leq s_{(n)}$

### 4.4 Prediction

For new query $q$:

1. Generate prediction $\hat{a}$
2. Compute threshold: $\hat{q} = s_{(\lceil(n+1)(1-\alpha)\rceil)}$
3. **Prediction is reliable if** estimated nonconformity $< \hat{q}$

### 4.5 Coverage Guarantee

**Theorem** [Vovk, 2005]: For exchangeable data:
$$P(s(q_{n+1}, a^*_{n+1}) \leq \hat{q}) \geq 1 - \alpha$$

This provides a **finite-sample, distribution-free guarantee**.

## 5. Combined Uncertainty

We combine retrieval and generation uncertainty:

### 5.1 Uncertainty Aggregation

$$U_{total} = \lambda_1 U_{retrieval} + \lambda_2 U_{MC} + \lambda_3 (1 - \text{conf}_{conformal})$$

where $\lambda_1 + \lambda_2 + \lambda_3 = 1$.

### 5.2 Confidence Calibration

Final confidence is calibrated using:
$$\text{conf}_{calibrated} = \sigma\left(\frac{\text{conf}_{raw} - \mu}{\tau}\right)$$

where $\mu$ and $\tau$ are learned on validation set.

## 6. Evaluation Metrics

### 6.1 Expected Calibration Error (ECE)

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} |\text{acc}(B_m) - \text{conf}(B_m)|$$

where $B_m$ are confidence bins, $\text{acc}(B_m)$ is accuracy in bin $m$.

### 6.2 Maximum Calibration Error (MCE)

$$\text{MCE} = \max_m |\text{acc}(B_m) - \text{conf}(B_m)|$$

### 6.3 Brier Score

$$\text{Brier} = \frac{1}{n}\sum_{i=1}^n (\text{conf}_i - \mathbb{1}[\text{correct}_i])^2$$

## 7. Novel Contributions

1. **First UQ framework for RAG**: Prior work focuses on LLM-only uncertainty
2. **Retrieval-generation uncertainty coupling**: Joint modeling of both sources
3. **Differentiable routing**: End-to-end learnable retrieval strategy selection
4. **Conformal guarantees for RAG**: Distribution-free coverage for retrieval-augmented systems

## 8. Theoretical Analysis

### 8.1 Retrieval Error Propagation

**Proposition**: Let $\epsilon_r$ be retrieval error (probability of missing relevant doc). Then:
$$P(\text{correct answer}) \leq (1 - \epsilon_r) \cdot P(\text{correct} | \text{relevant doc})$$

This shows that retrieval quality is a **hard ceiling** on answer quality.

### 8.2 Router Optimality

**Theorem**: Under the assumption that relevance is binary and known, the optimal router weight is:
$$w^*(p) = \mathbb{1}[s_{dense}(p) > s_{BM25}(p) \text{ for relevant } p]$$

Our learned router approximates this oracle.

## 9. References

1. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. ICML.

2. Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic learning in a random world. Springer.

3. Qin, T., Liu, T. Y., & Li, H. (2010). A general approximation framework for direct optimization of information retrieval measures. Information Retrieval.

4. Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS.

5. Shafer, G., & Vovk, V. (2008). A tutorial on conformal prediction. Journal of Machine Learning Research.
