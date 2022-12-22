#%%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

def makedist(df, figsize, cols, nrows, ncols, y=False):
    if y == False:
        plt.figure(figsize=figsize)
        for idx, col in enumerate(cols):
            plt.subplot(nrows, ncols, idx+1)
            sns.distplot(df[cols])
            plt.title(f'{col}')
            plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=figsize)
        for idx, col in enumerate(cols):
            plt.subplot(nrows, ncols, idx+1)
            sns.distplot(df[cols], df[y])
            plt.title(f'{col}')
            plt.tight_layout()
        plt.show()

def makehist(df, figsize, cols, nrows, ncols , y=False):
    if y == False:
        plt.figure(figsize=figsize)
        for idx, col in enumerate(cols):
            plt.subplot(nrows, ncols,idx+1)
            sns.histplot(df, x=col)
            plt.title(f'{col}')
            plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=figsize)
        for idx, col in enumerate(cols):
            plt.subplot(nrows, ncols,idx+1)
            sns.histplot(df, x=col, y=y)
            plt.title(f'{col}')
            plt.tight_layout()
        plt.show()
# %%
