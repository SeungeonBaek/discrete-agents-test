from typing import List, Dict, Union, Any
from numpy.typing import NDArray

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import GroupNormalization


class FlattenExtractor(Model):
    def __init__(self, )-> None:
        super(FlattenExtractor,self).__init__()

        self.feature = Flatten()

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = self.feature(state)

        return feature


class MLPExtractor(Model):
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(MLPExtractor,self).__init__()

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
        self.net_arc = self.config.get('network_architecture', [])
        if self.config.get('use_norm', False) == True:
            self.net_list = [None for _ in self.net_arc*2]
        else:
            self.net_list = [None for _ in self.net_arc]

        # Define the network architecture
        if self.config.get('use_norm', False) == True:
            for idx in range(0, len(self.net_arc*2), 2):
                self.net_list[idx] = Dense(self.net_arc[int(idx/2)], activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

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

        self.feature = Dense(feature_dim, activation = self.config.get('act_fn', 'relu'))

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        for idx, net in enumerate(self.net_list):
            if idx == 0:
                hidden = net(state)
            else:
                hidden = net(hidden)

        feature = self.feature(hidden)

        return feature


if __name__ == "__main__":
    # Test required
    pass