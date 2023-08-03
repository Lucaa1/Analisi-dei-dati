import numpy as np
import  matplotlib.pyplot  as plt
from sklearn.metrics import ConfusionMatrixDisplay

'''
Script in cui ci sono le funzioni per fare i grafici
'''

#Grafico confusion Matrix 
def plot_confusion_matrix(conf_matrix, classes=[0,1,2], title='Confusion matrix'):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
    disp.plot(cmap=plt.cm.Greens, values_format='.0f')
    plt.title(title)
    plt.show()

# Grafico delle metriche singole
def plot_metrics(m_avg, M_avg, title='m_avg and M_avg Metrics'):
    metrics = ['Accuracy', 'Balanced Accuracy', 'F1-Score']
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, m_avg, label='m_avg', color='b', alpha=0.7)
    plt.bar(metrics, M_avg, label='M_avg', color='g', alpha=0.7)
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

#Grafico delle metriche raggruppate
def plot_grouped_bar_chart(data, labels, title='Comparison of m_avg Metrics', metrics = ['Accuracy', 'Balanced Accuracy', 'Sensitivity', 'Specificity', 'F1-Score']):
    
    n_methods = len(data)
    width = 0.2
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_methods):
        bars = ax.bar(x + (i - n_methods/2) * width, data[i], width, label=labels[i])

        # Aggiungi il valore sopra ogni barra
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('Value')
    ax.set_title(title)

    ax.set_xticks(x - width/2)  
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()