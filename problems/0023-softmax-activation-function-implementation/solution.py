import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    total = sum(exp_scores)
    return [x / total for x in exp_scores]