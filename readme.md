# Code Usage Instructions for *REAR*

This repository contains the official implementation of the paper:

> **REAR: A Reinforcement Learning Extractor and Refiner for Semi-Supervised Local Community Detection in Anti-Money Laundering**

## Requirements

[Conda](https://docs.conda.io/en/latest/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is **strongly recommended** for managing dependencies.  
While it is technically possible to use a plain Python virtual environment, manually installing and aligning all required packages can be cumbersome and is **not recommended** for reproducibility.
## Setup

### 1. Create and activate the Conda environment

```bash
conda env create -f rear.yml
conda activate rear
```

### 2. Set the working directory

All commands should be executed from the `codes` directory. Running the code from any other location may cause the data processing module to fail.

```bash
cd codes
```

### 3. Obtain the dataset

The dataset is provided as `df.zip` in the repository root.

> ⚠️ **Important:** `df.zip` is managed with **Git LFS**. If you cloned the repository without Git LFS installed, you will only have a small pointer file (a few hundred bytes), not the actual dataset.

**To obtain the real data, use one of the following methods:**

**Option A (recommended) — Pull via Git LFS**

```bash
git lfs install
git lfs pull
```

**Option B — Download manually from GitHub**

1. Navigate to the repository on GitHub.
2. Click on the `df.zip` file.
3. Click the **Download** button to save the actual file.

After obtaining the full `df.zip`, unzip it into the location expected by `components/nn/dataProcess.py`. If the default path does not match your setup, you may adjust the file path variable inside that module.

### 4. Run the code

```bash
python components/main.py
```

## Notes

- The code was originally designed for NPU and CUDA environments but has been adapted for **CPU-only execution**. Performance on CPU is acceptable for typical usage.
- If you encounter any issues locating the data, please check the path variable in `components/nn/dataProcess.py` and modify it to point to your actual data directory.
