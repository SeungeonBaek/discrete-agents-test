import traceback

from typing import List, Dict, Union, Any
from numpy.typing import NDArray

import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, LSTMCell
from tensorflow.keras.layers import GRU, GRUCell
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import GroupNormalization

import sys, os
if __name__ == "__main__":
	sys.path.append(os.getcwd())


class GNNExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(GNNExtractor,self).__init__()

        self.config = extractor_config

        # Initializer
        if self.config.get('initializer', None) == 'glorot_normal':
            self.initializer = initializers.glorot_normal()
        elif self.config.get('initializer', None) == 'he_normal':
            self.initializer = initializers.he_normal()
        elif self.config.get('initializer', None) == 'orthogonal':
            self.initializer = initializers.orthogonal()
        else:
            self.initializer = initializers.random_normal()

        # Regularizer
        if self.config.get('regularizer', None) == 'l1':
            self.regularizer = regularizers.l1(l=self.config['regularizer']['l1']) # 0.0005
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['regularizer']['l2']) # 0.0005
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['regularizer']['l1'], l2=self.config['regularizer']['l2']) # 0.0005, 0.0005
        else:
            self.regularizer = None

        # Loading the network architecture


        # Define the network architecture


        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:

        feature = self.feature(state)

        return feature


class GCNExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(GCNExtractor,self).__init__()

        self.config = extractor_config

        # Initializer
        if self.config.get('initializer', None) == 'glorot_normal':
            self.initializer = initializers.glorot_normal()
        elif self.config.get('initializer', None) == 'he_normal':
            self.initializer = initializers.he_normal()
        elif self.config.get('initializer', None) == 'orthogonal':
            self.initializer = initializers.orthogonal()
        else:
            self.initializer = initializers.random_normal()

        # Regularizer
        if self.config.get('regularizer', None) == 'l1':
            self.regularizer = regularizers.l1(l=self.config['regularizer']['l1']) # 0.0005
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['regularizer']['l2']) # 0.0005
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['regularizer']['l1'], l2=self.config['regularizer']['l2']) # 0.0005, 0.0005
        else:
            self.regularizer = None

        # Loading the network architecture


        # Define the network architecture


        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:

        feature = self.feature(state)

        return feature


class GATExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(GATExtractor,self).__init__()

        self.config = extractor_config

        # Initializer
        if self.config.get('initializer', None) == 'glorot_normal':
            self.initializer = initializers.glorot_normal()
        elif self.config.get('initializer', None) == 'he_normal':
            self.initializer = initializers.he_normal()
        elif self.config.get('initializer', None) == 'orthogonal':
            self.initializer = initializers.orthogonal()
        else:
            self.initializer = initializers.random_normal()

        # Regularizer
        if self.config.get('regularizer', None) == 'l1':
            self.regularizer = regularizers.l1(l=self.config['regularizer']['l1']) # 0.0005
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['regularizer']['l2']) # 0.0005
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['regularizer']['l1'], l2=self.config['regularizer']['l2']) # 0.0005, 0.0005
        else:
            self.regularizer = None

        # Loading the network architecture


        # Define the network architecture


        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:

        feature = self.feature(state)

        return feature


class GaANExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(GATExtractor,self).__init__()

        self.config = extractor_config

        # Initializer
        if self.config.get('initializer', None) == 'glorot_normal':
            self.initializer = initializers.glorot_normal()
        elif self.config.get('initializer', None) == 'he_normal':
            self.initializer = initializers.he_normal()
        elif self.config.get('initializer', None) == 'orthogonal':
            self.initializer = initializers.orthogonal()
        else:
            self.initializer = initializers.random_normal()

        # Regularizer
        if self.config.get('regularizer', None) == 'l1':
            self.regularizer = regularizers.l1(l=self.config['regularizer']['l1']) # 0.0005
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['regularizer']['l2']) # 0.0005
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['regularizer']['l1'], l2=self.config['regularizer']['l2']) # 0.0005, 0.0005
        else:
            self.regularizer = None

        # Loading the network architecture


        # Define the network architecture


        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:

        feature = self.feature(state)

        return feature


class ECCExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(ECCExtractor,self).__init__()

        self.config = extractor_config

        # Initializer
        if self.config.get('initializer', None) == 'glorot_normal':
            self.initializer = initializers.glorot_normal()
        elif self.config.get('initializer', None) == 'he_normal':
            self.initializer = initializers.he_normal()
        elif self.config.get('initializer', None) == 'orthogonal':
            self.initializer = initializers.orthogonal()
        else:
            self.initializer = initializers.random_normal()

        # Regularizer
        if self.config.get('regularizer', None) == 'l1':
            self.regularizer = regularizers.l1(l=self.config['regularizer']['l1']) # 0.0005
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['regularizer']['l2']) # 0.0005
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['regularizer']['l1'], l2=self.config['regularizer']['l2']) # 0.0005, 0.0005
        else:
            self.regularizer = None

        # Loading the network architecture


        # Define the network architecture


        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:

        feature = self.feature(state)

        return feature


class GINExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(GINExtractor,self).__init__()

        self.config = extractor_config

        # Initializer
        if self.config.get('initializer', None) == 'glorot_normal':
            self.initializer = initializers.glorot_normal()
        elif self.config.get('initializer', None) == 'he_normal':
            self.initializer = initializers.he_normal()
        elif self.config.get('initializer', None) == 'orthogonal':
            self.initializer = initializers.orthogonal()
        else:
            self.initializer = initializers.random_normal()

        # Regularizer
        if self.config.get('regularizer', None) == 'l1':
            self.regularizer = regularizers.l1(l=self.config['regularizer']['l1']) # 0.0005
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['regularizer']['l2']) # 0.0005
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['regularizer']['l1'], l2=self.config['regularizer']['l2']) # 0.0005, 0.0005
        else:
            self.regularizer = None

        # Loading the network architecture


        # Define the network architecture


        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:

        feature = self.feature(state)

        return feature


class GraphSageExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(GraphSageExtractor,self).__init__()

        self.config = extractor_config

        # Initializer
        if self.config.get('initializer', None) == 'glorot_normal':
            self.initializer = initializers.glorot_normal()
        elif self.config.get('initializer', None) == 'he_normal':
            self.initializer = initializers.he_normal()
        elif self.config.get('initializer', None) == 'orthogonal':
            self.initializer = initializers.orthogonal()
        else:
            self.initializer = initializers.random_normal()

        # Regularizer
        if self.config.get('regularizer', None) == 'l1':
            self.regularizer = regularizers.l1(l=self.config['regularizer']['l1']) # 0.0005
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['regularizer']['l2']) # 0.0005
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['regularizer']['l1'], l2=self.config['regularizer']['l2']) # 0.0005, 0.0005
        else:
            self.regularizer = None

        # Loading the network architecture


        # Define the network architecture


        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:

        feature = self.feature(state)

        return feature


class PinSageExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(PinSageExtractor,self).__init__()

        self.config = extractor_config

        # Initializer
        if self.config.get('initializer', None) == 'glorot_normal':
            self.initializer = initializers.glorot_normal()
        elif self.config.get('initializer', None) == 'he_normal':
            self.initializer = initializers.he_normal()
        elif self.config.get('initializer', None) == 'orthogonal':
            self.initializer = initializers.orthogonal()
        else:
            self.initializer = initializers.random_normal()

        # Regularizer
        if self.config.get('regularizer', None) == 'l1':
            self.regularizer = regularizers.l1(l=self.config['regularizer']['l1']) # 0.0005
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['regularizer']['l2']) # 0.0005
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['regularizer']['l1'], l2=self.config['regularizer']['l2']) # 0.0005, 0.0005
        else:
            self.regularizer = None

        # Loading the network architecture


        # Define the network architecture


        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:

        feature = self.feature(state)

        return feature


def load_GraphExtractor(extractor_config:Dict, feature_dim): # Todo
    if extractor_config.get('name', None) == 'GNN':
        return GNNExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'GCN':
        return GCNExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'GAT':
        return GATExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'GaAN':
        return GaANExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'ECC':
        return ECCExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'GIN':
        return GINExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'GraphSage':
        return GraphSageExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'PinSage':
        return PinSageExtractor(extractor_config, feature_dim)
    else:
        raise ValueError("please use the MLPExtractor in ['GNN', 'GCN', 'GIN', 'GraphSage', 'PinSage', 'GaAN',\
                          'GAT', 'GATv2', 'ECC', 'GINE', 'GraphTransformer', 'XENetConv', 'CensNetConv', 'CrystalConv']")

def test_GraphExtractor(test_switch:int)-> None: # Todo
    if test_switch == 1:
        from extractor_config import Graph_GNN_extractor_config as extractor_config, Graph_GNN_feature_dim as feature_dim
    elif test_switch == 2:
        from extractor_config import Graph_GCN_extractor_config, Graph_GCN_feature_dim
    elif test_switch == 3:
        from extractor_config import Graph_GAT_extractor_config, Graph_GAT_feature_dim
    elif test_switch == 4:
        pass
    elif test_switch == 5:
        pass
    elif test_switch == 6:
        pass
    elif test_switch == 7:
        pass
    elif test_switch == 8:
        pass
    elif test_switch == 9:
        pass
    elif test_switch == 10:
        pass
    elif test_switch == 11:
        pass
    elif test_switch == 12:
        pass
    elif test_switch == 13:
        pass
    elif test_switch == 14:
        pass
    else:
        raise ValueError("Please use the correct test switch")

    extractor = load_GraphExtractor(extractor_config, feature_dim)

    try:
        test = extractor(np.ones(shape=[128, 128]))
    except Exception as e:
        print(f"error: {e}")
        print(f"error: {traceback.format_exc()}")


if __name__ == "__main__":
    """
    Graph Extractor
    1: GNN Extractor,                2: GCN Extractor,
    3: GIN Extractor,                4: GraphSage Extractor,
    5: PinSage Extractor,            6: GaAN Extractor,
    7: GAT Extractor,                8: GATv2 Extractor,
    9: ECC Extractor,               10: GINE Extractor,
    11: GraphTransformer Extractor, 12: XENetConv Extractor,
    13: CensNetConv Extractor,      14: CrystalConv Extractor
    """

    test_switch = 2

    # Test any extractor
    test_GraphExtractor(test_switch)
