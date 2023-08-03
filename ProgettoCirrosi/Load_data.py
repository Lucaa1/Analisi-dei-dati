import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

'''
Funzione che ripulisce i dati e crea un nuovo file .csv con i dati puliti
'''


def load_data(Norm=True, Encoding=True, Missing_value='MM'):
    df = pd.read_csv('archive/cirrhosis.csv')
    
    #Missing value per feature        
    if False:
        missing_values_count = df.isnull().sum()

        plt.figure(figsize=(12, 6))
        plt.bar(missing_values_count.index, missing_values_count.values, color="#238b45", alpha=0.9)
        plt.title('Numero di Missing Values per ogni Feature', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()
    
    #Duplicati
    if False:   
        if df.duplicated().all() == False:
            print('Non ci sono duplicati\n')
        else:
            print('Ci sono duplicati')

    #Stage
    if False:
        plt.figure(figsize=(20,5))
        sns.countplot(y=df['Stage'],palette="Greens", alpha=0.8)
        plt.xlabel('Numero di pazienti', fontsize=16)
        plt.ylabel('Stage', fontsize=16)        
        plt.show()

    df=df.drop(['ID'], axis=1)
    df=df.dropna(subset=['Stage'])

    #Considero 'Stage' come dato non numerico
    Numerical_col=df.select_dtypes(include=['int64', 'float64']).columns
    Numerical_col=Numerical_col.drop('Stage')
    Non_numerical_col=df.columns.drop(Numerical_col)

    if Missing_value == 'MM':
        #Per i data di tipo numerico vediamo come sono distribuiti per decidere se sostituirli con mediana o media
        Missing_data_Num=df[Numerical_col].columns[df[Numerical_col].isna().any()]

        for c in Missing_data_Num:
            df[c].fillna(df[c].median(), inplace=True)

        Missing_data_Non_Num=df[Non_numerical_col].columns[df[Non_numerical_col].isna().any()]
        for c in Missing_data_Non_Num:
            df[c].fillna(df[c].mode().values[0], inplace=True)

    elif Missing_value == 'Distribuzione':
        for col in df.columns:
            if df[col].isna().any():
                nans=df[col].isna()
                existing_values=df[col].dropna()
                num_missing=df[col].isna().sum()
                sampled_values=np.random.choice(existing_values, num_missing)
                df.loc[nans, col]=sampled_values
    elif Missing_value == '':
        pass
    else:
        raise ValueError("Valore 'Missing_data' non esatto.")

    if Encoding:
        for a in df.select_dtypes(exclude=['int64', 'float64']):
            df[a] = LabelEncoder().fit_transform(df[a])

    df['Stage']=df['Stage'].astype(int)

    if Norm:
        col = df.columns.drop('Stage')
        df[col] = StandardScaler().fit_transform(df[col])

    target = df['Stage']
    features = df.drop(columns='Stage')

    if False:
        df.to_csv('archive/New_dataset.csv', index=False)
    return df, features, target

# df, features, target = load_data(Norm=True, Encoding=True, Missing_value='MM')
