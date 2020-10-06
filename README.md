# Heart dynamics conditioned on brain

Code related to the paper:

*Joint data imputation and mechanistic modelling for simulating heart-brain interactions in incomplete datasets*

by Banus, Jaume and Sermesant, Maxime and Camara, Oscar and Lorenzi, Marco.

Link: 

BibTeX citation:
```bibtex
@article{Banus2020,
author = {Banus, Jaume and Sermesant, Maxime and Camara, Oscar and Lorenzi, Marco},
booktitle = {MICCAI},
title = {Joint data imputation and mechanistic modelling for simulating heart-brain interactions in incomplete datasets. Medical Image Computing and Computer Assisted Intervention},
year = {2020}
}
```

# Installation

1. Create/activate a virtual environment

- Install conda: 
# GNU / Linux
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
bash Miniconda3-latest-Linux-x86_64.sh
```

Download this github repository and move into in:
```bash
git clone https://gitlab.inria.fr/epione/hdcob
cd hdcob
```

# Windows
Download and install conda from: https://docs.conda.io/en/latest/miniconda.html
Download this github repository from: https://gitlab.inria.fr/epione/hdcob
Open the Anaconda prompt and move into the github repository previously downloaded.

2. Dependencies

- Install the customized python environment:
```bash
conda env create -f environment.yml
```

Activate the python environment:
```bash
conda activate py37
```

 - Or install the dependencies listed in 'requirements.txt'

3. Package installation, when we are inside the folder "hdcob"

- Using pip
```bash
pip install -e .
```

- Manually

Install the hdcob package:
```bash
python setup.py install
```

An alternative to the last point is to install the package in "develop" mode.
Using this mode, all local modifications of source code will be considered in your Python interpreter (when restarted) without having to install the package again.
This is particularly useful when adding new features.
To install this package in develop mode, type the following command line:
```bash
python setup.py develop
```



