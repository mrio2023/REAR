**Code Usage Instructions for *REAR: A Reinforcement Learning Extractor and Refiner for Semi-Supervised Local Community Detection in Anti-Money Laundering***

To use this code, please ensure that Conda or Miniconda is installed on your system.

1. **Create and activate the environment**  
   Use the provided `rear.yml` file to set up the environment:
   ```bash
   conda env create -f rear.yml
   conda activate rear
   ```

2. **Set the correct working directory**  
   The code expects the root directory to be `codes`. Running the code from a different location may cause the data processing logic to fail. Please verify that your current working directory is set to `codes` before execution.

3. **Run the code**  
   Execute the main script `components/main.py` (e.g., by running `python components/main.py` or using your IDE's run functionality).

> **Note**: The code was originally designed for NPU and CUDA, but the logic has been simplified to support CPU-only execution. Running on CPU yields acceptable performance for typical use cases.

The data processing module is located at `components/nn/dataProcess.py`. If the code fails to locate the required data, you can modify the file path variable in that module to point to your actual data directory.
