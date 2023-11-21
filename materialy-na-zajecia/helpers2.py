import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score


def plotROCs(results):
    ''' Funkcja, do rysowania szeregu wyników krzywych ROC dla poszczególnych eksperymentów
    results - lista wyników jako 3 elementowe tuple (true, pred, label)
    '''
    
    # Ustalanie wielkości rysunku
    fig, ax = plt.subplots(figsize=(5,4.5))
        
    for true, pred, label in results:
        # Obliczenie punktów potrzebnych do narysowani akrzywej ROC
        # funkcja roc_curve zwarca trzy serie danych, fpr, tpr oraz poziomy progów odcięcia
        fpr, tpr, thresholds = roc_curve(true, pred)
        # Obliczamy pole powierzchni pod krzywą
        rocScore = roc_auc_score(true, pred)
        rocScore = round(rocScore, 3)
        # Grubość krzywej
        lw = 2
        # Rysujemy krzywą ROC
        ax.plot(fpr, tpr, lw=lw, label=f'{label}: {rocScore}')
    # Rysujemy krzywą 45 stopni jako punkt odniesienia
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # Dodajemy lekkie marginesy do zakresu aby krzywa nie pokrywała się z osiami
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.show()
    
    
def plotPrRecalls(results):
    ''' Funkcja, do rysowania szeregu wyników krzywych PrRecall dla poszczególnych eksperymentów
    results - lista wyników jako 3 elementowe tuple (true, pred, label)
    '''
    
    # Ustalanie wielkości rysunku
    fig, ax = plt.subplots(figsize=(5,4.5))
    
    for true, pred, label in results:
        # Liczymy punkty potrzebne do narysowania krzywej
        precisionYes, recallYes, thresholdsYes = precision_recall_curve(true, pred)
        average_precisionYes = np.round(average_precision_score(true, pred), 4)
        # Rysujemy krzywą PR-Recall
        ax.plot(recallYes, precisionYes, color='r', lw=2, label=f'{label}: {average_precisionYes}')
    # Dodajemy lekkie marginesy do zakresu aby krzywa nie pokrywała się z osiami
    ax.set_xlim([0, 1])
    ax.set_ylim([0.0, 1.01])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc="lower left")
    plt.show()

def plotROCandPRs(results):
    ''' Funkcja, do rysowania szeregu wyników krzywych ROC i Precision recal dla poszczególnych eksperymentów
    results - lista wyników jako 3 elementowe tuple (true, pred, label)
    '''
    
    # Ustalanie wielkości rysunku
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
    for true, pred, label in results:
        # Obliczenie punktów potrzebnych do narysowania krzywej ROC
        fpr, tpr, thresholds = roc_curve(true, pred)
        # Obliczamy pole powierzchni pod krzywą
        rocScore = round(roc_auc_score(true, pred), 3)
        # Rysujemy krzywą ROC
        axes[0].plot(fpr, tpr, lw=2, label=f'{label}: {rocScore}')
        # Liczymy punkty potrzebne do narysowania krzywej PR
        precisionYes, recallYes, thresholdsYes = precision_recall_curve(true, pred)
        # Obliczamy pole powierzchni pod krzywą
        average_precisionYes = np.round(average_precision_score(true, pred), 4)
        # Rysujemy krzywą PR-Recall
        axes[1].plot(recallYes, precisionYes, lw=2, label=f'{label}: {average_precisionYes}')
        
    # Formatowanie wykresu ROC
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([-0.01, 1.0])
    axes[0].set_ylim([0.0, 1.01])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'Receiver operating characteristic')
    axes[0].legend(loc="lower right")

    # Formatowanie wykresu PR
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0.0, 1.01])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc="lower left")