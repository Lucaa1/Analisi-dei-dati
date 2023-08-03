import pandas as pd
import matplotlib.pyplot as plt
import Load_data as ld
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import warnings


'''
In questo script ci sono le funzioni usate per fare i grafici per la parte di analisi dei dati
'''


# Ignora i messaggi di warning specifici
warnings.filterwarnings("ignore", category=FutureWarning, message="Setting a gradient palette using color= is deprecated and will be removed in version 0.13. Set `palette='dark:red'` for same effect.")

df, features_df, target_df=ld.load_data(Norm=True, Encoding=True, Missing_value='Distribuzione')

# Correlazioni
def correlation(df, title=''):
    correlation_matrix = df.corr(method='pearson')  
    plt.figure(figsize=(10,10))
    sns.heatmap(correlation_matrix, vmin=-1, vmax=1, center=0,cmap=sns.diverging_palette(20, 220, n=200), square=True,annot=True, fmt=".2f", annot_kws={"size": 10})
    plt.title(title, fontsize=16)
    
    return correlation_matrix
# correlation(df, 'Correlazioni originali')


# Densità delle feauteres
def plot_densita(df):
    selected_features = [col for col in df.columns if df[col].nunique() > 4]
    n_rows=3
    n_col= math.ceil(np.shape(selected_features)[0]/n_rows)  #arrotondo per eccesso

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_col, figsize=(15, 8))
    for i, feature in enumerate(selected_features):
        axes[i // n_col, i % n_col].set_ylabel('Valori', fontsize=14)
        axes[i // n_col, i % n_col].set_xlabel(feature, fontsize=14)
        sns.histplot(data=df[feature], bins=20, kde=True, ax=axes[i // n_col, i % n_col],color="#69b3a2")
    plt.tight_layout()
# plot_densita(df)


# Individuo gli outliers
def plot_with_outliers(df, z_score=3):
    #Seleziono solo le feature che hanno più di 4 valori diversi
    selected_features = [col for col in df.columns if df[col].nunique() > 4]
    n_col = math.ceil(np.shape(selected_features)[0] / 2)  # arrotondo per eccesso

    scaler = StandardScaler()
    df_zscore = pd.DataFrame(scaler.fit_transform(df[selected_features]), columns=selected_features)

    #Visualizzo i boxplot
    fig, axes = plt.subplots(nrows=2, ncols=n_col, figsize=(15, 8))
    Outliers = []
    mat1 = []
    mat2 = []
    for i, feature in enumerate(selected_features):
        sns.boxplot(data=df_zscore[feature], ax=axes[i // n_col, i % n_col], boxprops=dict(facecolor="#69b3a2"))

        outliers = df_zscore[np.abs(df_zscore[feature]) > z_score]
        outliers = outliers.index.tolist()
        Outliers.extend(outliers)
        mat1.append(np.shape(outliers)[0])
        mat2.append(feature)

        axes[i // n_col, i % n_col].set_title(feature)

    Mat = [mat2, mat1]
    plt.tight_layout()

    #Grafico a barre del numero degli outliers 
    plt.figure(figsize=(10, 6))
    sns.barplot(y=mat2, x=mat1, color="#69b3a2", edgecolor='black', alpha=0.9)
    plt.xlabel('Numero di outliers')
    plt.ylabel('Feature')
    plt.title('Numero di outliers per ogni feature')

    return np.array(sorted(Outliers)), Mat
# plot_with_outliers(df)

plt.show()
