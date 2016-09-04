# pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv("data/train.csv", dtype={"Age": np.float64}, )


test_df = pd.read_csv("data/test.csv", dtype={"Age": np.float64}, )