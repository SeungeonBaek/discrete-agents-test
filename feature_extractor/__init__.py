from typing import Dict

from MLPExtractor import *
from ConvolutionalExtractor import *
from RecurrentExtractor import *
from AttentionExtractor import *
from GraphExtractor import *
from CustomExtractor import *


def define_extractor(extractor_config: Dict):
    if extractor_config.get("type", None) == "MLP":
        load_MLPExtractor(extractor_config, extractor_config.get("feature_dim", 128))

    elif extractor_config.get("type", None) == "Convolutional":
        load_ConvExtractor(extractor_config, extractor_config.get("feature_dim", 128))

    elif extractor_config.get("type", None) == "Recurrent":
        load_RecurExtractor(extractor_config, extractor_config.get("feature_dim", 128))

    elif extractor_config.get("type", None) == "Attention":
        load_AttentionExtractor(extractor_config, extractor_config.get("feature_dim", 128))

    elif extractor_config.get("type", None) == "Graph":
        load_GraphExtractor(extractor_config, extractor_config.get("feature_dim", 128))

    elif extractor_config.get("type", None) == "Custom":
        load_CustomExtractor(extractor_config, extractor_config.get("feature_dim", 128))
        
    else:
        raise ValueError("please use correct extractor type in ['MLP', 'Convolutional', 'Recurrent', 'Attention', 'Graph', 'Custom']")


if __name__ == "__main__":
    # elu, gelu, linear, relu, selu, sigmoid, softmax, softplus, swish, tanh
    pass