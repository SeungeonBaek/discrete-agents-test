from typing import Dict

import numpy as np

if __name__ == "__main__":
    from MLPExtractor import *
    from ConvolutionalExtractor import *
    from RecurrentExtractor import *
    from AttentionExtractor import *
    from GraphExtractor import *
    from CustomExtractor import *
else:
    from feature_extractor.MLPExtractor import *
    from feature_extractor.ConvolutionalExtractor import *
    from feature_extractor.RecurrentExtractor import *
    from feature_extractor.AttentionExtractor import *
    from feature_extractor.GraphExtractor import *
    from feature_extractor.CustomExtractor import *


def define_extractor(extractor_config: Dict):
    if extractor_config.get("type", None) == "MLP":
        return load_MLPExtractor(extractor_config, extractor_config.get("feature_dim", 128))

    elif extractor_config.get("type", None) == "Convolutional":
        return load_ConvExtractor(extractor_config, extractor_config.get("feature_dim", 128))

    elif extractor_config.get("type", None) == "Recurrent":
        return load_RecurExtractor(extractor_config, extractor_config.get("feature_dim", 128))

    elif extractor_config.get("type", None) == "Attention":
        return load_AttentionExtractor(extractor_config, extractor_config.get("feature_dim", 128))

    elif extractor_config.get("type", None) == "Graph":
        return load_GraphExtractor(extractor_config, extractor_config.get("feature_dim", 128))

    elif extractor_config.get("type", None) == "Custom":
        return load_CustomExtractor(extractor_config, extractor_config.get("feature_dim", 128))
        
    else:
        raise ValueError("please use correct extractor type in ['MLP', 'Convolutional', 'Recurrent', 'Attention', 'Graph', 'Custom']")


def test_extractor(extractor_config, feature_dim, test_input):
    extractor_config['feature_dim'] = feature_dim
    
    extractor = define_extractor(extractor_config)
    try:
        test = extractor(test_input)
    except Exception as e:
        print(f"error: {e}")
        print(f"error: {traceback.format_exc()}")


if __name__ == "__main__":
    from extractor_config import *

    """
    Extractor
    1: MLP Extractor,       2: Convolutional Extractor, 3: Recurrent Extractor,
    4: Attention Extractor, 5: Graph Extractor,         6: Custom Extractor
    """

    test_switch = 1

    # Test any extractor
    if test_switch == 1:
        test_extractor(MLP_flatten_extractor_config, MLP_flatten_feature_dim, np.ones(shape=(128,128)))

    elif test_switch == 2:
        test_extractor(Convolutional_CNN1d_extractor_config, Convolutional_CNN1d_feature_dim, None)

    elif test_switch == 3:
        test_extractor(REC_lstm_extractor_config, REC_lstm_feature_dim,None)

    elif test_switch == 4:
        test_extractor(Attention_attention_extractor_config, Attention_attention_feature_dim, None)

    elif test_switch == 5:
        test_extractor(Graph_GCN_extractor_config, Graph_GCN_feature_dim, None)

    elif test_switch == 6:
        test_extractor(Custom_simple_mlp_extractor_config, Custom_simple_mlp_feature_dim, None)
        
    else:
        raise ValueError("please use correct test switch in [1, 2, 3, 4, 5, 6]")