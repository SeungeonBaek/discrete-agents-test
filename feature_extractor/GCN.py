import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.keras import regularizers


class NodeEmbeddingLayer(Layer):
    def __init__(self, node_size, node_feature_size):
        super(NodeEmbeddingLayer, self).__init__()
        self.node_size = node_size
        self.node_feature_size = node_feature_size

        self.kernel_initializer = initializers.he_normal()
        self.kernel_regularizer = regularizers.l2(l=0.005)

    def build(self, ):
        super(NodeEmbeddingLayer, self).build(self.node_feature_size)

        self.kernel = self.add_weight(
            shape=(self.node_feature_size, self.node_feature_size),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

        self.b = self.add_weight(
            shape=(self.node_feature_size, ),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)
    
    def call(self, obs):
        # print(f'in node_embedding, obs : {obs.shape}')
        embedding = tf.einsum('bnf,ff->bnf',obs, self.kernel)
        # print(f'in node_embedding, embedding 1: {embedding.shape} \n {embedding}')
        embedding = tf.add(embedding, self.b)
        # print(f'in node_embedding, embedding 3: {embedding.shape} \n {self.b} \n {embedding}')

        return embedding


class ReadOutLayer(Layer):
    def __init__(self, node_size, node_feature_size, read_out_size):
        super(ReadOutLayer, self).__init__()
        self.node_size = node_size
        self.node_feature_size = node_feature_size
        self.read_out_size = read_out_size

        self.kernel_initializer = initializers.he_normal()
        self.kernel_regularizer = regularizers.l2(l=0.005)

    def build(self, ):
        super(ReadOutLayer, self).build([self.node_size, self.node_feature_size, self.read_out_size])

        self.feature_kernel = self.add_weight(
            shape=(self.node_feature_size, ),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

        self.feature_bias = self.add_weight(
            shape=(self.node_size, ),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

        self.read_out_kernel = self.add_weight(
            shape=(self.node_size, self.read_out_size),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

        self.read_out_bias = self.add_weight(
            shape=(self.read_out_size, ),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True)

    def call(self, embedding):
        # print(f'in read_out, embedding 0 : {embedding.shape}')
        read_out = tf.einsum('bnf,f->bn', embedding, self.feature_kernel)
        # print(f'in read_out, read_out 1: {read_out.shape}')
        read_out = tf.add(read_out, self.feature_bias)
        # print(f'in read_out, read_out 2: {read_out.shape}')
        read_out = tf.nn.relu(read_out)
        # print(f'in read_out, read_out 3: {read_out.shape}')

        read_out = tf.einsum('bn,nd->bd', read_out, self.read_out_kernel)
        # print(f'in read_out, read_out 4: {read_out.shape}')
        read_out = tf.add(read_out, self.read_out_bias)
        # print(f'in read_out, read_out 5: {read_out.shape}')
        read_out = tf.nn.relu(read_out)
        # print(f'in read_out, read_out 6: {read_out.shape}')

        return read_out


class GCNEmbedding(Model):
    '''
    This GCN consider graph data without edge features
    '''
    def __init__(self, node_size, node_feature_size, read_out_size):
        super(GCNEmbedding,self).__init__()
        self.node_size = node_size
        self.node_feature_size = node_feature_size
        self.adjacency_matrix_size = (node_size, node_size)        

        self.embedding1 = NodeEmbeddingLayer(self.node_size, self.node_feature_size)
        self.embedding1.build()
        self.embedding2 = NodeEmbeddingLayer(self.node_size, self.node_feature_size)
        self.embedding2.build()

        self.read_out = ReadOutLayer(node_size, node_feature_size, read_out_size)
        self.read_out.build()

    def call(self, state, adj_matrix):
        embedding1 = self.embedding1(state)
        embedding1 = tf.einsum('nn, bnf->bnf', adj_matrix, embedding1)
        embedding1 = tf.nn.relu(embedding1)

        embedding2 = self.embedding2(embedding1)
        embedding2 = tf.einsum('nn, bnf->bnf', adj_matrix, embedding2)
        embedding2 = tf.nn.relu(embedding2)

        read_out = self.read_out(embedding2)

        return read_out


if __name__ == '__main__':
    test_gcn = GCNEmbedding(node_size=4, node_feature_size=3, read_out_size=5)
    data = [[[1.1, 1.2, 2.0],
             [0.0, 1.0, 3.3],
             [0.0, 0.0, 5.5],
             [3.0, 2.4, 1.0]]]
    
    adj_matrix = [[1, 1, 1, 0],
                  [1, 1, 0, 1],
                  [1, 0, 1, 1],
                  [0, 1, 1, 1]]
    
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    adj_matrix = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)
    processed_data = test_gcn(data, adj_matrix)
    print(processed_data.shape)