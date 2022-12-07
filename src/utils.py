import torch
import math


def create_mid_split_binary_mask(features):
    """
    Creates a binary mask of a given dimension which splits its masking at the midpoint.
    :param features: Dimension of mask.
    :return: Binary mask split at midpoint of type torch.Tensor
    """
    mask = torch.zeros(features).byte()
    midpoint = features // 2 if features % 2 == 0 else features // 2 + 1
    mask[:midpoint] += 1
    return mask

def nats_to_bits_per_dim(nats, c, h, w):
    return nats / (math.log(2) * c * h * w)