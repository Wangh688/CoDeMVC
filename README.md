# CoDe-MVC: Counterfactual Debiased Multi-View Clustering via Causal Disentanglement

> **Authors (co-first):**
> :contentReference[oaicite:0]{index=0}, :contentReference[oaicite:1]{index=1}

This repository contains the official PyTorch implementation of **CoDe-MVC**, a counterfactual debiased deep multi-view clustering framework based on **structural causal modeling**, **content–style disentanglement**, and **counterfactual invariance learning**.

- Paper: **Submitted / Under review** (link will be released after submission)
- Code: :contentReference[oaicite:2]{index=2} repository of CoDe-MVC
- Contact: **1337326302@qq.com**
- Pretrained models: **Not provided** (training from scratch)

---

## 1. Overview

<img src="figures/pipeline.png" width="900" />

**Pipeline of CoDe-MVC.**  
Stage 1 learns causal disentangled representations (content/style).  
Stage 2 performs counterfactual style intervention (mixup/shuffle) and causal reweighting to suppress nuisance agreement.  
Stage 3 fuses debiased content across views and performs clustering.

> **Note:**  
> If you prefer vector quality for reviewers, replace the image with `figures/pipeline.pdf` and change the tag to:
> `<img src="figures/pipeline.pdf" width="900" />`

---

## 2. Requirements

We recommend the following environment (consistent with our experiments):

- python >= 3.10  
- pytorch >= 2.0 (we used 2.5.1)  
- numpy, scipy, scikit-learn  
- (optional) CUDA for GPU training

### Install
```bash
# (recommended) create an environment
conda create -n codemvc python=3.10 -y
conda activate codemvc

# install dependencies (once requirements.txt is provided)
pip install -r requirements.txt