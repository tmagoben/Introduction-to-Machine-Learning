import numpy as np

import scipy as sp

import pandas as pd

import matplotlib.pyplot as plt

### This is the second chapter of Andrew Ng's Machine Learning Course
# We begin with logistic regresion, and plotting decision boundaries

## Here we are loading the data, while the first two columns contain
# The exam scores, the third contains the labels
# As this is labeled data, we are dealing with supervised learning

data = pd.read_csv('ex2data1.txt')
