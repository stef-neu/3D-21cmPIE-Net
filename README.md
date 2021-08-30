# 3D-21cmPIE-Net

Repository for reproducing the results from [Arxiv Link]

Installation
------------
1. Install [21cmFAST][21cmFAST]
2. Install [tf-keras-vis][tf-keras-vis]
3. Clone this repository
    ```
    git clone https://github.com/stef-neu/3D-21cmPIE-Net.git
    cd 3D-21cmPIE-Net
    ```

4. Install the required dependencies

  - with `conda`:
    ```
    conda env create -f environment.yml
    conda activate 3D-21cmPIE-Net
    ```
  - with `pip`:
    ``` 
    python3 -m pip install -r requirements/requirements.txt
    ```

5. Install `3D-21cmPIE-Net`

  ```
  python3 -m pip install .
  ```

[21cmFAST]: https://github.com/21cmfast/21cmFAST
[tf-keras-vis]: https://github.com/keisen/tf-keras-vis

Usage
-----
1. Produce a dataset in `simulations` with a size of N lightcones. It is recommended to not produce large datasets with a single run. runSimulations.py creates a new output file for each run.
    ```
    runSimulations.py [options] [N_lightcones]
    ```
2. Train the neural network in `3DCNN`. By default this takes all output files from runSimulations.py.
    ```
    runCNN.py [options] [epochs]
    ```
The trained 3D CNNs and test results from the paper can be found under paper_results. The code for the paper plots can be found in paper_plots. This code uses the data from paper_results.

Achknowledgement
----------------
If you use any part of this repository please cite the following paper:
```text
[Bibtex Citation]
```
