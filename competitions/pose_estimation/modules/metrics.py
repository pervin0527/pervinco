"""
"""
import torch
def get_metric_fn(y_pred, y_answer):
    """ Metric 함수 반환하는 함수

    Returns:
        metric_fn (Callable)
    """
    mpjpe = torch.pow(torch.cat(y_pred) - torch.cat(y_answer), 2).sum(dim=2).mean(dim=1).mean().item()
    return mpjpe

