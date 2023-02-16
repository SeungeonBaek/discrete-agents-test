from typing import List, Dict, Union, Any
from numpy.typing import NDArray

import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import GroupNormalization


class FlattenExtractor(Model):
    """
    Flatten feature extractor

    Argument:
        extractor_config: agent configuration which is realted with RL algorithm => QR-DQN
            {
                name: name of feature extractor
                initializer: initializer of final layers. ex) 'glorot_normal'
                regularizer: regularizer of final layers. ex) 'l1_l2'
                    'l1': value of l1 coefficient if user choose regularizer as 'l1' or 'l1_l2'
                    'l2': value of l2 coefficient if user choose regularizer as 'l2' or 'l1_l2'
            }

        feature_dim: shpae of feature. ex) 128

    Properties:
        config: extractor configuration
        initializer: initializer of whole layers
        regularizer: regularizer of whole layers
        feature: final layer

    """
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(FlattenExtractor,self).__init__()

        self.config = extractor_config
        self.name = self.config['name']

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
            self.regularizer = regularizers.l1(l=self.config['l1']) # 0.0005
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['l2']) # 0.0005
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['l1'], l2=self.config['l2']) # 0.0005, 0.0005
        else:
            self.regularizer = None

        self.flatten = Flatten()
        self.feature = Dense(feature_dim, activation = 'linear', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        flatten = self.flatten(state)
        feature = self.feature(flatten)

        return feature


class MLPExtractor(Model):
    """
    MLP extractor which could be configured by user

    Argument:
        extractor_config: agent configuration which is realted with RL algorithm => QR-DQN
            {
                name: name of feature extractor
                initializer: initializer of whole layers. ex) 'glorot_normal'
                regularizer: regularizer of whole layers. ex) 'l1_l2'
                    'l1': value of l1 coefficient if user choose regularizer as 'l1' or 'l1_l2'
                    'l2': value of l2 coefficient if user choose regularizer as 'l2' or 'l1_l2'
                network_architecture: network_architecture of MLP. ex) [256, 256]
                use_norm: indicator whether the usage of normalization layer. ex) True
                norm_type: type of normalization layer. ex) 'layer_norm'
                act_fn: activation function of whole layers. ex) 'relu'
            }

        feature_dim: shpae of feature. ex) 128

    Properties:
        config: extractor configuration
        initializer: initializer of whole layers
        regularizer: regularizer of whole layers
        net_arc: architecture of network
        net_list: list of layers in whole network
        feature: final layer

    """
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(MLPExtractor,self).__init__()

        self.config = extractor_config
        self.name = self.config['name']

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
            self.regularizer = regularizers.l1(l=self.config['l1']) # 0.0005
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['l2']) # 0.0005
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['l1'], l2=self.config['l2']) # 0.0005, 0.0005
        else:
            self.regularizer = None

        # Loading the network architecture
        self.net_arc = self.config.get('network_architecture', [])
        if self.config.get('use_norm', False) == True:
            self.net_list = [None for _ in self.net_arc*2]
        else:
            self.net_list = [None for _ in self.net_arc]

        # Define the network architecture
        if self.config.get('use_norm', False) == True:
            for idx in range(0, len(self.net_arc*2), 2):
                self.net_list[idx] = Dense(self.net_arc[int(idx/2)], activation = self.config.get('act_fn', 'relu'), kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

                if self.config.get('norm_type', None) == "layer_norm":
                    self.net_list[idx+1] = LayerNormalization(axis=-1)
                elif self.config.get('norm_type', None) == "batch_norm":
                    self.net_list[idx+1] = BatchNormalization(axis=-1)
                elif self.config.get('norm_type', None) == "group_norm":
                    raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
                else:
                    raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', batch_norm', ']")
        else:
            for idx, node_num in enumerate(self.net_arc):
                self.net_list[idx] = Dense(node_num, activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

        self.feature = Dense(feature_dim, activation = 'linear', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        for idx, net in enumerate(self.net_list):
            if idx == 0:
                hidden = net(state)
            else:
                hidden = net(hidden)

        feature = self.feature(hidden)

        return feature


def load_MLPExtractor(extractor_config:Dict, feature_dim: int)-> Model:
    if extractor_config.get('name', None) == 'Flatten':
        return FlattenExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'MLP':
        return MLPExtractor(extractor_config, feature_dim)
    else:
        raise ValueError("please use the MLPExtractor in ['Flatten', 'MLP']")


if __name__ == "__main__":
    # For Flatten extractor
    flatten_extractor_config = {'name': 'Flatten', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005}
    flatten_feature_dim = 128

    flatten_extractor = load_MLPExtractor(flatten_extractor_config, flatten_feature_dim)
    test = flatten_extractor(np.ones(shape=[128, 128]))

    # For MLP extractor
    mlp_extractor_config = {'name': 'MLP', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                            'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
    mlp_feature_dim = 128

    mlp_extractor = load_MLPExtractor(mlp_extractor_config, mlp_feature_dim)
    test = mlp_extractor(np.ones(shape=[128, 128]))
