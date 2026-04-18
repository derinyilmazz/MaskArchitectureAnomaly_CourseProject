import numpy as np

def softmax_np(logits, axis=0):
    """Numerically stable softmax."""
    exp_logits = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

def msp_anomaly_score(logits):
    """Max Softmax Probability (MSP). Higher score = more anomalous."""
    probs = softmax_np(logits, axis=0)
    max_probs = np.max(probs, axis=0)
    return 1.0 - max_probs

def maxlogit_anomaly_score(logits):
    """Max Logit anomaly score. Higher score = more anomalous."""
    max_logits = np.max(logits, axis=0)
    return -max_logits

def entropy_anomaly_score(logits):
    """Max Entropy anomaly score. Higher score = more anomalous."""
    probs = softmax_np(logits, axis=0)
    epsilon = 1e-10 # Add small epsilon to avoid log(0)
    entropy = -np.sum(probs * np.log(probs + epsilon), axis=0)
    return entropy