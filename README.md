This repository contains the code to generate the figures of the manuscript:
Louisiane Lemaire, Mathieu Desroches, Serafim Rodrigues, Fabien Campillo. [Depolarization block induction via slow Na<sup>V</sup>1.1 inactivation in Dravet syndrome](https://arxiv.org/abs/2505.03919), 2025.

## Prerequisites
- The continuation software [AUTO-07p](https://github.com/auto-07p/auto-07p)
- Anaconda or Miniconda. Alternatively, you may install the dependencies using any method you prefer (see what is needed in [environment.yml](environment.yml)).

## How to use

1. Clone this repository

2. Create a virtual environment with the dependencies.
```commandline
conda env create -f environment.yml
```

3. Activate the environment and start the jupyter notebook server
```commandline
conda activate slow-nav-inactivation-in-dravet-syndrome
jupyter notebook
```

4. There is one notebook for each figure. Run them to generate the figures. Note that it can take some time. Figures can then be found in the "figures" directory.