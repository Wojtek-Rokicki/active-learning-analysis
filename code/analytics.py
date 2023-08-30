# Analyse dataframe in the context of features' values distribution
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def analyse_dataset(df: pd.DataFrame) -> None:
    print(f'Dataset {df.name}:\n'+10*'-')
    print("Describe:")
    print(df.describe())
    print("Classes:")
    targets = df.target.value_counts()
    targets.sort_values(inplace=True)
    print(targets)
    print(f"IR: {targets.iloc[1]/targets.iloc[0]:.2f}")

    nrows = math.ceil(len(df.columns)/3)
    fig, axes = plt.subplots(nrows = nrows, ncols = 3)
    axes = axes.flatten()
    fig.set_size_inches(15, nrows*5)
    fig.suptitle(f'Dataset \"{df.name}\" features\' values distributions', fontsize=24)

    for ax, col in zip(axes, df.columns):
        sns.histplot(df[col], ax = ax,  kde=True)
        ax.set(xlabel=f'Feature \"{col}\" value')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

from sklearn.decomposition import PCA
RANDOM_STATE_SEED = 13
