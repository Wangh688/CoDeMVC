## CoDe-MVC: Counterfactual Debiased Multi-View Clustering via Causal Disentanglement

> **Authors (co-first):**
>
> Dexun Zhao, Hao Wang, 

This repository contains the official PyTorch implementation of **CoDe-MVC**, a counterfactual debiased deep multi-view clustering framework based on **structural causal modeling**, **content–style disentanglement**, and **counterfactual invariance learning**.

- Contact: **1337326302@qq.com**
- Pretrained models: **Not provided** (training from scratch)

---

## 1. Overview

<img src="figures/pipeline.png" width="900" />

**Pipeline of CoDe-MVC.**  
Stage 1 learns causal disentangled representations (content/style).  
Stage 2 performs counterfactual style intervention (mixup/shuffle) and causal reweighting to suppress nuisance agreement.  
Stage 3 fuses debiased content across views and performs clustering.

---

## 2. Requirements

### Tested environment
- OS: Windows
- Python: 3.10
- PyTorch: 2.5.1
- CUDA: 12.1 (optional, for GPU training)

> Note: The above versions are the tested setup used in our experiments.

### Python dependencies
- numpy
- scipy
- scikit-learn

### Install
```bash
conda create -n codemvc python=3.10 -y
conda activate codemvc

pip install -r requirements.txt

```
---

## 3. Datasets

- The all datasets could be downloaded from [cloud](https://pan.baidu.com/s/1HCUQtvkLo-vv_GZ9aXKz0Q?pwd=6688). key: 6688

---

## 4. Usage

### To train a new model, run:

```bash
python train.py
```

---

## 5. Experiment Results

<img src="https://github.com/Wangh688/CoDeMVC/blob/main/figures/results.png"  width="900"  />

---

## 6. Acknowledgments

Our proposed CoDeMVC are inspired by [SCMVC](), [CausalMVC](), and [CauMVC](). Thanks for these valuable works.