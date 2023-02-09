from typing import Dict

from MLPExtractor import *
from ConvolutionalExtractor import *
from RecurrentExtractor import *
from AttentionExtractor import *
from GraphExtractor import *
from CustomExtractor import *


def define_extractor(extractor_config: Dict):
    if extractor_config.get("type", None) == "MLP":
        pass
    elif extractor_config.get("type", None) == "Convolutional":
        pass
    elif extractor_config.get("type", None) == "Recurrent":
        pass
    elif extractor_config.get("type", None) == "Attention":
        pass
    elif extractor_config.get("type", None) == "Graph":
        pass
    elif extractor_config.get("type", None) == "Custom":
        pass
    else:
        raise ValueError("please use correct extractor type in ['MLP', 'Convolutional', 'Recurrent', 'Attention', 'Graph', 'Custom']")


if __name__ == "__main__":
    # elu, gelu, linear, relu, selu, sigmoid, softmax, softplus, swish, tanh
    pass