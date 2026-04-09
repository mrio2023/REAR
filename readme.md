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

3. **Prepare the data**  
   The dataset is provided as `df.zip` in the root directory. Before running the code, **unzip this file** (e.g., using `unzip df.zip` or your preferred archive tool). Ensure the extracted contents are placed in the expected location as defined in `components/nn/dataProcess.py`.

4. **Run the code**  
   Execute the main script `components/main.py` (e.g., by running `python components/main.py` or using your IDE's run functionality).

> **Note**: The code was originally designed for NPU and CUDA, but the logic has been simplified to support CPU-only execution. Running on CPU yields acceptable performance for typical use cases.

The data processing module is located at `components/nn/dataProcess.py`. If the code fails to locate the required data, you can modify the file path variable in that module to point to your actual data directory.

