# Paper Implementation: AlphaSeek

Reference implementation of "AlphaSeek: model-agnostic sequential experimental design for materials research using conformal prediction" with scripts to reproduce experiments.

## Structure
- data/ — datasets used in the paper
- model/ — pretrained models used for polymer design
- examples/ — experiment scripts and configs
- utils.py — core utilities for the proposed method

## Requirements
- Python 3.12.11
- numpy 1.26.4, pandas 2.1.4, torch 2.9.0+cpu, scikit-learn 1.7.2, scipy 1.16.3
