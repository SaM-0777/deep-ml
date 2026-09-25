import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    probs = []
    for i in range(len(true_labels)):
        label = true_labels[i]
        correct_class_predicted_probs = predicted_probs[i][label == 1][0]
        probs.append(correct_class_predicted_probs + epsilon)
    return -np.mean(np.log(probs))