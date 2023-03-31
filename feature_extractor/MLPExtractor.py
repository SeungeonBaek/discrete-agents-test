import traceback

from typing import List, Dict, Union, Any, Tuple
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
# from tensorflow.keras.layers import GroupNormalizationz

import sys, os
if __name__ == "__main__":
	sys.path.append(os.getcwd())


class FlattenExtractor(Model):
    """
    Flatten feature extractor

    Argument:
        extractor_config: agent configuration which is realted with RL algorithm
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
        name: extractor name
        initializer: initializer of whole layers
        regularizer: regularizer of whole layers
        feature: final layer

    """
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(FlattenExtractor,self).__init__()

        self.config = extractor_config
        self.extractor_name = self.config['name']

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
        '''
        dim of state: (batch_size, states)
        '''
        flatten = self.flatten(state)
        feature = self.feature(flatten)

        return feature


class MLPExtractor(Model):
    """
    MLP extractor which could be configured by user

    Argument:
        extractor_config: agent configuration which is realted with RL algorithm
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
        name: extractor name
        initializer: initializer of whole layers
        regularizer: regularizer of whole layers
        net_arc: architecture of network
        net_list: list of layers in whole network
        feature: final layer

    """
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(MLPExtractor,self).__init__()

        self.config = extractor_config
        self.extractor_name = self.config['name']

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
            self.regularizer = regularizers.l1(l=self.config['l1'])
        elif self.config.get('regularizer', None) == 'l2':
            self.regularizer = regularizers.l2(l=self.config['l2'])
        elif self.config.get('regularizer', None) == 'l1_l2':
            self.regularizer = regularizers.l1_l2(l1=self.config['l1'], l2=self.config['l2'])
        else:
            self.regularizer = None

        # Loading the network architecture
        self.net_arc = self.config.get('network_architecture', [256, 256])
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
                    raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")
        else:
            for idx, node_num in enumerate(self.net_arc):
                self.net_list[idx] = Dense(node_num, activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

        self.feature = Dense(feature_dim, activation = 'linear', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        '''
        dim of state: (batch_size, states)
        '''
        for idx, net in enumerate(self.net_list):
            if idx == 0:
                hidden = net(state)
            else:
                hidden = net(hidden)

        feature = self.feature(hidden)

        return feature


class AutoEncoder1DExtractor(Model): # Todo
    """
    AutoEncoder with MLP extractor which could be configured by user

    Argument:
        extractor_config: agent configuration which is realted with RL algorithm
            {
                name: name of feature extractor
                initializer: initializer of whole layers. ex) 'glorot_normal'
                regularizer: regularizer of whole layers. ex) 'l1_l2'
                    'l1': value of l1 coefficient if user choose regularizer as 'l1' or 'l1_l2'
                    'l2': value of l2 coefficient if user choose regularizer as 'l2' or 'l1_l2'
                network_architecture: network_architecture of UNet.
                    ex) [[256, 128], 64, [128, 256]]
                use_norm: indicator whether the usage of normalization layer. ex) True
                norm_type: type of normalization layer. ex) 'layer_norm'
                act_fn: activation function of whole layers. ex) 'relu'
            }

        feature_dim: shpae of feature. ex) 128

    Properties:
        config: extractor configuration
        name: extractor name
        initializer: initializer of whole layers
        regularizer: regularizer of whole layers
        net_arc: architecture of network
        contract_net_list: list of contraction layers
        code_net: code network
        cond_net_n: normalization layer for code network
        expand_net_list: list of exapnsion layers

    Concept:
        256 => 128 => 64 => 128 => 256

    """
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(AutoEncoder1DExtractor,self).__init__()

        self.config = extractor_config
        self.extractor_name = self.config['name']
        self.ae_lr = self.config['lr']

        # Initializer
        self.initializer = initializers.glorot_normal()

        # Regularizer
        self.regularizer = regularizers.l2(l=0.0005)

        # Loading and defining the network architecture
        self.net_arc = self.config.get('network_architecture', [[256, 128], 64, [128, 256]])
        self.act_fn = self.config.get('act_fn', 'relu')
        self.contract_net_list = []
        self.code_net, self.code_net_n = None, None
        self.expand_net_list = []

        # Define the network architecture
        # Contraction
        for inner_node in self.net_arc[0]:
            self.contract_net_list.append(Dense(inner_node, activation = self.act_fn, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer))

            if self.config.get('norm_type', None) == "layer_norm":
                self.contract_net_list.append(LayerNormalization(axis=-1))
            elif self.config.get('norm_type', None) == "batch_norm":
                self.contract_net_list.append(BatchNormalization(axis=-1))
            elif self.config.get('norm_type', None) == "group_norm":
                raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
            else:
                raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")

        # Code
        self.code_net = Dense(self.net_arc[1], activation = self.act_fn, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

        if self.config.get('norm_type', None) == "layer_norm":
            self.code_net_n = LayerNormalization(axis=-1)
        elif self.config.get('norm_type', None) == "batch_norm":
            self.code_net_n = BatchNormalization(axis=-1)
        elif self.config.get('norm_type', None) == "group_norm":
            raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
        else:
            raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")

        # Expansion
        for idx, inner_node in enumerate(self.net_arc[-1]):
            self.expand_net_list.append(Dense(inner_node, activation = self.act_fn, kernel_initializer=self.initializer, kernel_regularizer=self.regularizer))

            if self.config.get('norm_type', None) == "layer_norm":
                self.expand_net_list.append(LayerNormalization(axis=-1))
            elif self.config.get('norm_type', None) == "batch_norm":
                self.expand_net_list.append(BatchNormalization(axis=-1))
            elif self.config.get('norm_type', None) == "group_norm":
                raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
            else:
                raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")

        self.reconstruct = Dense(self.config.get('reconstruct_dim', 8), activation = 'linear', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

    def call(self, state: Union[NDArray, tf.Tensor])-> Tuple[tf.Tensor]:
        '''
        dim of state: (batch_size, states)
        '''
        # Contraction
        for idx, net in enumerate(self.contract_net_list):
            if idx == 0:
                hidden = net(state)
            else:
                hidden = net(hidden)

        # Code
        code = self.code_net(hidden)
        code = self.code_net_n(code)

        # Expansion
        for idx, net in enumerate(self.expand_net_list):
            if idx == 0:
                hidden = net(code)
            else:
                hidden = net(hidden)

        reconstruct = self.reconstruct(hidden)

        return code, reconstruct


class Inception1DExtractor(Model):
    """
    Inception with MLP extractor which could be configured by user

    Argument:
        extractor_config: agent configuration which is realted with RL algorithm
            {
                name: name of feature extractor
                initializer: initializer of whole layers. ex) 'glorot_normal'
                regularizer: regularizer of whole layers. ex) 'l1_l2'
                    'l1': value of l1 coefficient if user choose regularizer as 'l1' or 'l1_l2'
                    'l2': value of l2 coefficient if user choose regularizer as 'l2' or 'l1_l2'
                network_architecture: network_architecture of Inception. ex) [256, [128 64], 256]
                use_norm: indicator whether the usage of normalization layer. ex) True
                norm_type: type of normalization layer. ex) 'layer_norm'
                act_fn: activation function of whole layers. ex) 'relu'
            }

        feature_dim: shpae of feature. ex) 128

    Properties:
        config: extractor configuration
        name: extractor name
        initializer: initializer of whole layers
        regularizer: regularizer of whole layers
        net_arc: architecture of network
        net_list: list of layers in whole network
        inner_net_list: list of layers in inner network
        feature: final layer

    """
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(Inception1DExtractor,self).__init__()

        self.config = extractor_config
        self.extractor_name = self.config['name']

        # Initializer
        self.initializer = initializers.glorot_normal()

        # Regularizer
        self.regularizer = regularizers.l2(l=0.0005)

        # Loading the network architecture
        self.net_arc = self.config.get('network_architecture', [256, [128, 64, 32], 256])
        self.net_list = []
        self.inner_net_list = []

        # Define the network architecture
        for idx, net_info in enumerate(self.net_arc):
            if isinstance(net_info, int):
                self.net_list.append(Dense(net_info, activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer))
                if self.config.get('use_norm', False) == True:
                    if self.config.get('norm_type', None) == "layer_norm":
                        self.net_list.append(LayerNormalization(axis=-1))
                    elif self.config.get('norm_type', None) == "batch_norm":
                        self.net_list.append(BatchNormalization(axis=-1))
                    elif self.config.get('norm_type', None) == "group_norm":
                        raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
                    else:
                        raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")

            elif isinstance(net_info, List):
                for inner_idx, inner_net_info in enumerate(net_info):
                    self.inner_net_list.append(Dense(inner_net_info, activation = self.config.get('act_fn', 'relu') , kernel_initializer=self.initializer, kernel_regularizer=self.regularizer))
                    if self.config.get('use_norm', False) == True:
                        if self.config.get('norm_type', None) == "layer_norm":
                            self.net_list.append(LayerNormalization(axis=-1))
                        elif self.config.get('norm_type', None) == "batch_norm":
                            self.net_list.append(BatchNormalization(axis=-1))
                        elif self.config.get('norm_type', None) == "group_norm":
                            raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
                        else:
                            raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")

                self.net_list.append(self.inner_net_list)

            else:
                raise RuntimeError()

        self.feature = Dense(feature_dim, activation = 'linear', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        '''
        dim of state: (batch_size, states)
        '''
        for idx, net in enumerate(self.net_list):
            if idx == 0:
                hidden = net(state)
            else:
                if isinstance(net, Model):
                    hidden = net(hidden)
                elif isinstance(net, List):
                    for inner_net in net:
                        hidden = inner_net(hidden)

        feature = self.feature(hidden)

        return feature


class UNet1DExtractor(Model):
    """
    Unet with MLP extractor which could be configured by user

    Argument:
        extractor_config: agent configuration which is realted with RL algorithm
            {
                name: name of feature extractor
                initializer: initializer of whole layers. ex) 'glorot_normal'
                regularizer: regularizer of whole layers. ex) 'l1_l2'
                    'l1': value of l1 coefficient if user choose regularizer as 'l1' or 'l1_l2'
                    'l2': value of l2 coefficient if user choose regularizer as 'l2' or 'l1_l2'
                network_architecture: network_architecture of UNet.
                    ex) [[[128, 128], [64,64], [32,32]], 16, [[16, 32], [32, 64], [64,128]]]
                use_norm: indicator whether the usage of normalization layer. ex) True
                norm_type: type of normalization layer. ex) 'layer_norm'
                act_fn: activation function of whole layers. ex) 'relu'
            }

        feature_dim: shpae of feature. ex) 128

    Properties:
        config: extractor configuration
        name: extractor name
        initializer: initializer of whole layers
        regularizer: regularizer of whole layers
        net_arc: architecture of network
        contract_net_list: list of contraction layers
        code_net: code network
        expand_net_list: list of exapnsion layers
        act_fn: activation function of each layer
        feature: final layer

    Concept:
        128, 128                  =>                  64 + 64, 128       
                =>64, 64          =>         32 + 32, 64
                        =>32, 32  =>  16+16, 32
                                  =>  16

    """
    def __init__(self, extractor_config: Dict, feature_dim: int)-> None:
        super(UNet1DExtractor,self).__init__()

        self.config = extractor_config
        self.extractor_name = self.config['name']

        # Initializer
        self.initializer = initializers.glorot_normal()

        # Regularizer
        self.regularizer = regularizers.l2(l=0.0005)

        # Loading the network architecture and activation function
        self.net_arc = self.config.get('network_architecture', [[[128, 128], [64,64], [32,32]], 16, [[16, 32], [32, 64], [64,128]]])
        self.contract_net_list = []
        self.code_net = None
        self.expand_net_list = []

        if self.config.get('act_fn', None) == 'relu':
            self.act_fn = tf.nn.relu
        elif self.config.get('act_fn', None) == 'tanh':
            self.act_fn = tf.nn.tanh
        elif self.config.get('act_fn', None) == 'leaky_relu':
            self.act_fn = tf.nn.leaky_relu
        elif self.config.get('act_fn', None) == 'selu':
            self.act_fn = tf.nn.selu
        elif self.config.get('act_fn', None) == 'celu':
            self.act_fn = tf.nn.celu
        elif self.config.get('act_fn', None) == 'elu':
            self.act_fn = tf.nn.elu
        elif self.config.get('act_fn', None) == 'gelu':
            self.act_fn = tf.nn.gelu
        elif self.config.get('act_fn', None) == 'swish':
            self.act_fn = tf.nn.swish
        else:
            raise ValueError("Please check the activation function you used")

        # Define the network architecture
        # Contraction
        for inner_arc in self.net_arc[0]:
            for inner_node in inner_arc:
                self.contract_net_list.append(Dense(inner_node, activation = 'linear', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer))

                if self.config.get('norm_type', None) == "layer_norm":
                    self.contract_net_list.append(LayerNormalization(axis=-1))
                elif self.config.get('norm_type', None) == "batch_norm":
                    self.contract_net_list.append(BatchNormalization(axis=-1))
                elif self.config.get('norm_type', None) == "group_norm":
                    raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
                else:
                    raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")

        # Code
        self.code_net = Dense(self.net_arc[1], activation = 'linear', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

        # Expansion
        for inner_arc in self.net_arc[-1]:
            for inner_node in inner_arc:
                self.expand_net_list.append(Dense(inner_node, activation = 'linear', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer))

                if self.config.get('norm_type', None) == "layer_norm":
                    self.expand_net_list.append(LayerNormalization(axis=-1))
                elif self.config.get('norm_type', None) == "batch_norm":
                    self.expand_net_list.append(BatchNormalization(axis=-1))
                elif self.config.get('norm_type', None) == "group_norm":
                    raise ValueError("In this version (tf 2.7), group_norm layer is not supported")
                else:
                    raise ValueError("If you don't use the normalization, you might use 'use_norm == False' in extractor config. Or please check the norm_type in ['layer norm', 'batch_norm']")

        self.feature = Dense(feature_dim, activation = 'linear', kernel_initializer=self.initializer, kernel_regularizer=self.regularizer)

    def call(self, state: Union[NDArray, tf.Tensor])-> tf.Tensor:
        '''
        dim of state: (batch_size, states)
        '''
        jump = []
        # Contraction
        for idx, net in enumerate(self.contract_net_list):
            if idx == 0:
                hidden = net(state)
            else:
                if idx % 2 == 0:
                    hidden = net(hidden)
                else:
                    hidden = self.act_fn(net(hidden))
                    if idx % 4 == 3:
                        jump.append(hidden)

        # Code
        code = self.code_net(hidden)

        # Expansion
        for idx, net in enumerate(self.expand_net_list):
            if idx == 0:
                hidden = net(jump[-1])
                hidden = tf.concat((hidden, code), axis=1)
            else:
                if idx % 2 == 0:
                    if idx % 4 == 0:
                        transferred = net(jump[-(int(idx//4)+1)])
                        hidden = tf.concat((hidden, transferred), axis=1)
                    else:
                        hidden = net(hidden)
                else:
                    hidden = self.act_fn(net(hidden))

        feature = self.feature(hidden)

        return feature


def load_MLPExtractor(extractor_config:Dict, feature_dim: int)-> Model:
    if extractor_config.get('name', None) == 'Flatten':
        return FlattenExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'MLP':
        return MLPExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'AutoEncoder1D':
        return AutoEncoder1DExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'Inception1D':
        return Inception1DExtractor(extractor_config, feature_dim)
    elif extractor_config.get('name', None) == 'UNet1D':
        return UNet1DExtractor(extractor_config, feature_dim)
    else:
        raise ValueError("please use the MLPExtractor in ['Flatten', 'MLP', 'AutoEncoder1D', 'Inception1D']")


def test_MLPExtractor(extractor_config:Dict, feature_dim:int)-> None:
    extractor = load_MLPExtractor(extractor_config, feature_dim)

    try:
        if extractor_config["name"] == "AutoEncoder1D":
            test, reconst = extractor(np.ones(shape=[128, 8]))
            print(f'shape of test tensor: {test.shape}')
            print(f'shape of reconst tensor: {reconst.shape}')
        else:
            test = extractor(np.ones(shape=[128, 8]))
            print(f'shape of test tensor: {test.shape}')
    except Exception as e:
        print(f"error: {e}")
        print(f"error: {traceback.format_exc()}")


if __name__ == "__main__":
    from extractor_config import MLP_Flatten_extractor_config, MLP_Flatten_feature_dim
    from extractor_config import MLP_MLP_extractor_config, MLP_MLP_feature_dim
    from extractor_config import MLP_AE1d_extractor_config, MLP_AE1d_feature_dim
    from extractor_config import MLP_Inception1d_extractor_config, MLP_Inception1d_feature_dim
    from extractor_config import MLP_UNet1d_extractor_config, MLP_UNet1d_feature_dim

    """
    MLP Extractor
    1: Flatten Extractor, 2: MLP Extractor
    3: AE1D Extractor,    4: Inception1D Extractor
    5: UNet1D Extractor
    """

    test_switch = 3

    # Test any extractor
    if test_switch == 1:
        test_MLPExtractor(MLP_Flatten_extractor_config, MLP_Flatten_feature_dim)
    elif test_switch == 2:
        test_MLPExtractor(MLP_MLP_extractor_config, MLP_MLP_feature_dim)
    elif test_switch == 3:
        test_MLPExtractor(MLP_AE1d_extractor_config, MLP_AE1d_feature_dim)
    elif test_switch == 4:
        test_MLPExtractor(MLP_Inception1d_extractor_config, MLP_Inception1d_feature_dim)
    elif test_switch == 5:
        test_MLPExtractor(MLP_UNet1d_extractor_config, MLP_UNet1d_feature_dim)
    else:
        raise ValueError("Please correct the test switch in [1, 2, 3, 4, 5]")
