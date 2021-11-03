# 3D-21cmPIE-Net

'3D 21cm Parameter InfErence Network'

Repository for reproducing the results from [Arxiv Link].

3D-21cmPIE-Net is trained to jointly infer astrophysical and (warm) dark matter parameters, as well as astrophysical parameters alone, from 3D lightcones of the 21cm signal during reionization and the cosmic dawn. This repository also provides further functionalities such as an interface to 21cmFAST for lightcone creation (incl. mock observed lightcones for noise levels derived by 21cmSense), for network training and evaluation, as well as for network interpretation via saliency maps.  

Installation
------------
1. Clone this repository
    ```
    git clone https://github.com/stef-neu/3D-21cmPIE-Net.git
    cd 3D-21cmPIE-Net
    ```

2. Install the required dependencies (installation with conda is recommended)

  - with `conda`:
    ```
    conda env create -f environment.yml
    conda activate 3D-21cmPIE-Net
    ```
  - with `pip`:
    ``` 
    python3 -m pip install -r requirements/requirements.txt
    ```

    Note: For installation on MacOS (less tested, tensorflow CPU), please replace 'tensorflow-gpu' by 'tensorflow' in environment.yml and setup.py, or for pip-based installation in requirements/requirements.txt

3. Install `3D-21cmPIE-Net`

  ```
  python3 -m pip install .
  ```
  
4. When using `runSimulations.py`, `create_mocks.py` or `SaliencyMaps.py` install [21cmFAST][21cmFAST] 3.1.2

5. When using `SaliencyMaps.py` install [tf-keras-vis][tf-keras-vis]

[21cmFAST]: https://github.com/21cmfast/21cmFAST
[tf-keras-vis]: https://github.com/keisen/tf-keras-vis
[21cmSense]: https://github.com/jpober/21cmSense

Usage
-----
1. Produce a dataset by default saved to `simulations/output` with a size of N lightcones. It is recommended to not produce large datasets with a single run. runSimulations.py creates a new output file for each run.
    ```
    cd simulations
    python runSimulations.py [options] [N_lightcones]
    ```
    Folders 'simulations/output' and 'simulations/_cache' are created. 
    
A. Train the 3D CNN on bare simulations:
   - Train the neural network in `3DCNN`. By default this takes all output files from runSimulations.py as input.
       ```
       cd 3DCNN
       python runCNN.py [options] [epochs]
       ```
        
B: Train the 3D CNN on opt mocks:
  - Create opt mocks in `mock_creation`.
    ```
    python create_mocks.py [options]
    ```
  - Train the neural network in `3DCNN`. Here the input files have to be specified.
    ```
    python runCNN.py --data=../mock_creation/output/*.tfrecord [options] [epochs]
    ```
The trained 3D CNNs and test results from the paper are stored in paper_results. The code to reproduce the plots from the paper can be found under paper_plots. This code uses the data from paper_results.

Acknowledgements
-----------------
If you use any part of this repository please cite the following paper:
```text
[Bibtex Citation]
```
When using `runSimulations.py`, `create_mocks.py` or `SaliencyMaps.py` please also cite both of the following papers to ackknowledge [21cmFAST][21cmFAST].
    
```
@article{Murray_2020,
  doi = {10.21105/joss.02582},
  url = {https://doi.org/10.21105/joss.02582},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {54},
  pages = {2582},
  author = {Steven G. Murray and Bradley Greig and Andrei Mesinger and Julian B. Muñoz and Yuxiang Qin and Jaehong Park and Catherine A. Watkinson},
  title = {21cmFAST v3: A Python-integrated C code for generating 3D realizations of the cosmic 21cm signal.},
  journal = {Journal of Open Source Software}
}
```

```
@article{Mesinger_2011,
    author = {Mesinger, Andrei and Furlanetto, Steven and Cen, Renyue},
    title = "{21cmfast: a fast, seminumerical simulation of the high-redshift 21-cm signal}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {411},
    number = {2},
    pages = {955-972},
    year = {2011},
    month = {02},
    issn = {0035-8711},
    doi = {10.1111/j.1365-2966.2010.17731.x},
    url = {https://doi.org/10.1111/j.1365-2966.2010.17731.x},
    eprint = {https://academic.oup.com/mnras/article-pdf/411/2/955/4099991/mnras0411-0955.pdf},
}
```
When using any file in `mock_creation` please also cite the following papers and provide a link to the [21cmSense][21cmSense] repository.

```
@article{Pober_2013,
   title={THE BARYON ACOUSTIC OSCILLATION BROADBAND AND BROAD-BEAM ARRAY: DESIGN OVERVIEW AND SENSITIVITY FORECASTS},
   volume={145},
   ISSN={1538-3881},
   url={http://dx.doi.org/10.1088/0004-6256/145/3/65},
   DOI={10.1088/0004-6256/145/3/65},
   number={3},
   journal={The Astronomical Journal},
   publisher={American Astronomical Society},
   author={Pober, Jonathan C. and Parsons, Aaron R. and DeBoer, David R. and McDonald, Patrick and McQuinn, Matthew and Aguirre, James E. and Ali, Zaki and Bradley, Richard F. and Chang, Tzu-Ching and Morales, Miguel F.},
   year={2013},
   month={Jan},
   pages={65}
}
```

```
@article{Pober_2014,
   title={WHAT NEXT-GENERATION 21 cm POWER SPECTRUM MEASUREMENTS CAN TEACH US ABOUT THE EPOCH OF REIONIZATION},
   volume={782},
   ISSN={1538-4357},
   url={http://dx.doi.org/10.1088/0004-637X/782/2/66},
   DOI={10.1088/0004-637x/782/2/66},
   number={2},
   journal={The Astrophysical Journal},
   publisher={American Astronomical Society},
   author={Pober, Jonathan C. and Liu, Adrian and Dillon, Joshua S. and Aguirre, James E. and Bowman, Judd D. and Bradley, Richard F. and Carilli, Chris L. and DeBoer, David R. and Hewitt, Jacqueline N. and Jacobs, Daniel C. and et al.},
   year={2014},
   month={Jan},
   pages={66}
}
```
