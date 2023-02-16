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


# Example
class SimpleMLPExtractor(Model):
    """
    SimpleMLP extractor which could be configured by user

    Argument:
        extractor_config: agent configuration which is realted with RL algorithm => QR-DQN
            {
                name: name of feature extractor
                network_architecture: network_architecture of MLP. ex) [256, 256]
                activation_function: activation function of whole layers. ex) 'relu'
            }

        feature_dim: shpae of feature. ex) 128

    Properties:
        config: extractor configuration
        initializer: initializer of whole layers
        regularizer: regularizer of whole layers
        l1: first layer
        l1_n: layer normalization layer following first layer
        l2: second layer
        l2_n: layer normalization layer following second layer
        feature: final feature layer

    """
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(SimpleMLPExtractor,self).__init__()

        self.config = extractor_config
        self.name = self.config['name']

        # Initializer
        self.initializer = initializers.glorot_normal()

        # Regularizer
        self.regularizer = regularizers.l2(l=0.0005)

        # Loading and defining the network architecture
        self.net_arc = self.config.get('network_architecture', [256, 256])
        self.act_fn = self.config.get('activation_function', 'relu')

        self.l1 = Dense(self.net_arc[0], activation=self.act_fn, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l1_n = LayerNormalization(axis=-1)

        self.l2 = Dense(self.net_arc[1], activation=self.act_fn, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2_n = LayerNormalization(axis=-1)

        self.feature = Dense(feature_dim, activation = 'linear', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        '''
        dim of state: (batch_size, states)
        '''
        l1 = self.l1(state)
        l1_n = self.l1_n(l1)

        l2 = self.l2(l1_n)
        l2_n = self.l2_n(l2)

        feature = self.feature(l2_n)

        return feature


class DivineConquerExtractor(Model):
    """
    DivineConquer extractor which could be configured by user

    Argument:
        extractor_config: agent configuration which is realted with RL algorithm => QR-DQN
            {
                name: name of feature extractor
                network_architecture: network_architecture of MLP. ex) [128, [128, 128], 256]
                activation_function: activation function of whole layers. ex) 'relu'
            }

        feature_dim: shpae of feature. ex) 128

    Properties:
        config: extractor configuration
        initializer: initializer of whole layers
        regularizer: regularizer of whole layers
        l1: first level layer
        l1_n: layer normalization layer following l1 layer

            l2_1: first of second level layer
            l2_1n: layer normalization layer following l2_1 layer
            l2_2: second of second level layer
            l2_2n: layer normalization layer following l2_2 layer

        l3: third level layer
        l3_n: layer normalization layer following l3 layer

        feature: final feature layer

    """
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(DivineConquerExtractor,self).__init__()

        self.config = extractor_config
        self.name = self.config['name']

        # Initializer
        self.initializer = initializers.glorot_normal()

        # Regularizer
        self.regularizer = regularizers.l2(l=0.0005)

        # Loading and defining the network architecture
        self.net_arc = self.config.get('network_architecture', [128, [128, 128], 256])
        self.act_fn = self.config.get('activation_function', 'relu')

        self.l1 = Dense(self.net_arc[0], activation=self.act_fn, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l1_n = LayerNormalization(axis=-1)

        self.l2_1 = Dense(self.net_arc[1][0], activation=self.act_fn, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2_1n = LayerNormalization(axis=-1)
        self.l2_2 = Dense(self.net_arc[1][1], activation=self.act_fn, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l2_2n = LayerNormalization(axis=-1)

        self.l3 = Dense(self.net_arc[2], activation=self.act_fn, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)
        self.l3_n = LayerNormalization(axis=-1)

        self.feature = Dense(feature_dim, activation = 'linear', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        '''
        dim of state: (batch_size, states)
        '''
        l1 = self.l1(state)
        l1_n = self.l1_n(l1)

        l2_1 = self.l2_1(l1_n)
        l2_1n = self.l2_1n(l2_1)

        l2_2 = self.l2_2(l1_n)
        l2_2n = self.l2_2n(l2_2)

        l3 = self.l3(tf.concat([l2_1n, l2_2n], axis=1))

        feature = self.feature(l3)

        return feature


# Project 1
class ResidualExtractor(Model):
    def __init__(self, )-> None:
        super(ResidualExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


class AutoEncoderExtractor(Model):
    def __init__(self, )-> None:
        super(AutoEncoderExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


class UNetExtractor(Model):
    def __init__(self, )-> None:
        super(AutoEncoderExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


# Project 2
class LSTMExtractor(Model):
    def __init__(self, )-> None:
        super(LSTMExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


class CNN1DExtractor(Model):
    def __init__(self, )-> None:
        super(CNN1DExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


class BiLSTMExtractor(Model):
    def __init__(self, )-> None:
        super(BiLSTMExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


# Project 3
class AttentionExtractor(Model):
    def __init__(self, )-> None:
        super(AttentionExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


class TransDuctiveGNNExtractor(Model):
    def __init__(self, )-> None:
        super(TransDuctiveGNNExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


class InductiveGNNExtractor(Model):
    def __init__(self, )-> None:
        super(TransDuctiveGNNExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


class TransformerExtractor(Model):
    def __init__(self, )-> None:
        super(TransDuctiveGNNExtractor,self).__init__()

        pass

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        feature = None

        return feature


def load_CustomExtractor(extractor_config:Dict, feature_dim):
    if extractor_config.get('name', None) == 'ExamRes':
        return ResidualExtractor(extractor_config, feature_dim)

    elif extractor_config.get('name', None) == 'ExamAE':
        return AutoEncoderExtractor(extractor_config, feature_dim)

    elif extractor_config.get('name', None) == 'ExamUNet':
        return UNetExtractor(extractor_config, feature_dim)

    elif extractor_config.get('name', None) == 'ExamLSTM':
        return LSTMExtractor(extractor_config, feature_dim)

    elif extractor_config.get('name', None) == 'ExamCNN1D':
        return CNN1DExtractor(extractor_config, feature_dim)

    elif extractor_config.get('name', None) == 'ExamBiLSTM':
        return BiLSTMExtractor(extractor_config, feature_dim)

    elif extractor_config.get('name', None) == 'ExamAttention':
        return AttentionExtractor(extractor_config, feature_dim)

    elif extractor_config.get('name', None) == 'ExamTransductiveGNN':
        return TransDuctiveGNNExtractor(extractor_config, feature_dim)

    elif extractor_config.get('name', None) == 'ExamInductiveGNN':
        return InductiveGNNExtractor(extractor_config, feature_dim)

    elif extractor_config.get('name', None) == 'ExamTransformer':
        return TransformerExtractor(extractor_config, feature_dim)

    else:
        raise ValueError("please use the correct example extractor or modify the load func")


if __name__ == "__main__":
    pass