import numpy as np
from sklearn.metrics import roc_curve

def fpr_at_95_tpr(anomaly_scores, labels):
    """Calculates the False Positive Rate at 95% True Positive Rate."""
    fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)
    
    # Find the index where the True Positive Rate hits 95%
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        return fpr[idx[0]]
    else:
        return 1.0

# The original file imported these extra functions, 
# but evalAnomaly.py doesn't actually use them. 
# We put them here as "dummies" just to stop the import errors!
def calc_metrics(*args, **kwargs): pass
def plot_roc(*args, **kwargs): pass
def plot_pr(*args, **kwargs): pass
def plot_barcode(*args, **kwargs): pass