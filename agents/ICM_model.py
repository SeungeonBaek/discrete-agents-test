import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense


class ICM_feature(Model):
    def __init__(self, input_size, output_size):
        super(ICM_feature,self).__init__()
        self.kernel_initializer = initializers.he_normal()
        self.kernel_regularizer = regularizers.l2(l=0.005)

        self.target_layer1 = Dense(256, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.target_layer2 = Dense(256, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.target_output = Dense(output_size, activation = None)

    def call(self, next_state):
        layer1 = self.target_layer1(next_state) # 확인
        layer2 = self.target_layer2(layer1)
        target_output = self.target_output(layer2)

        return target_output

class Inverse_model(Model):
    def __init__(self, input_size, output_size):
        super(Inverse_model,self).__init__()
        self.kernel_initializer = initializers.he_normal()
        self.kernel_regularizer = regularizers.l1_l2(l1=0.01, l2=0.005)

        self.target_layer1 = Dense(256, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.target_layer2 = Dense(128, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.target_layer3 = Dense(64, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.target_layer4 = Dense(32, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.target_output = Dense(output_size, activation = None)

    def call(self, next_state):
        layer1 = self.target_layer1(next_state) # 확인
        layer2 = self.target_layer2(layer1)
        layer3 = self.target_layer3(layer2)
        layer4 = self.target_layer4(layer3)
        target_output = self.target_output(layer4)

        return target_output

class Forward_model(Model):
    def __init__(self, input_size, output_size):
        super(Forward_model,self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.kernel_initializer = initializers.he_normal()
        self.kernel_regularizer = regularizers.l1_l2(l1=0.01, l2=0.005)
        self.target_layer1 = Dense(256, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.target_layer2 = Dense(256, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.target_layer3 = Dense(128, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.target_layer4 = Dense(128, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.target_output = Dense(output_size, activation = None)

    def call(self, next_state):
        layer1 = self.target_layer1(next_state) # 확인
        layer2 = self.target_layer2(layer1)
        layer3 = self.target_layer3(layer2)
        layer4 = self.target_layer4(layer3)
        target_output = self.target_output(layer4)

        return target_output

class ICM_model(Model):
    def __init__(self, obs_size, action_size, feature_size):
        super(ICM_model, self).__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.feature_size = feature_size

        self.features = ICM_feature(self.obs_size, self.feature_size)
        self.forward_model = Forward_model(self.feature_size + self.action_size, self.feature_size)
        self.inverse_model = Inverse_model(self.feature_size + self.feature_size, self.action_size)

    def call(self, inputs:tuple):
        states, next_states, actions = inputs

        feature_states = self.features(states)
        feature_next_states = self.features(next_states)

        concat_feature_s_a = tf.concat((feature_states, actions), axis = -1)
        pred_feature_next_states = self.forward_model(concat_feature_s_a)

        concat_feature_s_next_s = tf.concat((feature_states, feature_next_states), axis = -1)
        pred_action = self.inverse_model(concat_feature_s_next_s)
        
        return feature_next_states, pred_feature_next_states, pred_action
