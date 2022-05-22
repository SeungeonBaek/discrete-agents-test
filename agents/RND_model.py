from tensorflow.keras import Model
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense

class RND_target(Model):
    def __init__(self, input_size, output_size):
        super(RND_target,self).__init__()
        self.kernel_initializer = initializers.orthogonal()
        self.kernel_regularizer = regularizers.l2(l=0.001)

        self.target_layer1 = Dense(256, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.target_layer2 = Dense(128, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.target_layer3 = Dense(64, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.target_layer4 = Dense(32, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.target_output = Dense(output_size, activation = None)

    def call(self, next_state):
        layer1 = self.target_layer1(next_state)
        layer2 = self.target_layer2(layer1)
        layer3 = self.target_layer3(layer2)
        layer4 = self.target_layer4(layer3)
        target_output = self.target_output(layer4)

        return target_output

class RND_predict(Model):
    def __init__(self, input_size, output_size):
        super(RND_predict,self).__init__()
        self.kernel_initializer = initializers.orthogonal()
        self.kernel_regularizer = regularizers.l2(l=0.001)

        self.predict_layer1 = Dense(256, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.predict_layer2 = Dense(128, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.predict_layer3 = Dense(128, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.predict_layer4 = Dense(64, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.predict_layer5 = Dense(64, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.predict_layer6 = Dense(32, activation = 'relu' , kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.predict_output = Dense(output_size, activation = None)

    def call(self, next_state):
        layer1 = self.predict_layer1(next_state) # 확인
        layer2 = self.predict_layer2(layer1)
        layer3 = self.predict_layer3(layer2)
        layer4 = self.predict_layer4(layer3)
        layer5 = self.predict_layer5(layer4)
        layer6 = self.predict_layer6(layer5)
        value = self.predict_output(layer6)

        return value