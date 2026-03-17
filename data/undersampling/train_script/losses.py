import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import torch.nn.functional as F

import os
_MASK_PATH = os.path.join(os.path.dirname(__file__), '../../safe_mask.npy')
if os.path.exists(_MASK_PATH):
    _safe_mask = torch.from_numpy(np.load(_MASK_PATH)).bool()  # shape (768, 1152)
    print(f'[losses.py] Masque géographique chargé — {_safe_mask.sum().item():,} pixels ignorés ({100*_safe_mask.float().mean().item():.1f}%)')
else:
    _safe_mask = None
    print('[losses.py] safe_mask.npy non trouvé — aucun masque appliqué')


def jaccard_loss(logits, true, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]

    #  Masque géographique
    if _safe_mask is not None:
        mask = _safe_mask.to(logits.device)
        # Applique le masque sur true : pixels masqués → classe 0 (Background)
        true = true.clone()
        true[:, mask] = 0

        # Applique le masque sur logits : pixels masqués → prédiction certaine Background
        logits = logits.clone()
        logits[:, 1:, mask] = -1e9   
        logits[:, 0, mask]  =  1e9  

    #  Calcul Jaccard standard 
    true_1_hot = torch.eye(num_classes).to(logits.device)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)