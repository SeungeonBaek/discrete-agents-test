#####################################################
############ MLP Extractor configuraiton ############
#####################################################

## For Flatten extractor
MLP_flatten_extractor_config = {'type': 'MLP', 'name': 'Flatten', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005}
MLP_flatten_feature_dim = 128

## For MLP extractor
MLP_mlp_extractor_config = {'type': 'MLP', 'name': 'MLP', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
MLP_mlp_feature_dim = 128


###########################################################
############ Recurrent Extractor configuraiton ############
###########################################################

## For LSTM extractor
REC_lstm_extractor_config = {'type': 'Recurrent', 'name': 'LSTM', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
REC_lstm_feature_dim = 128

## For LSTMCell extractor
REC_lstm_cell_extractor_config = {'type': 'Recurrent', 'name': 'LSTMCell', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
REC_lstm_cell_feature_dim = 128

## For GRU extractor
REC_gru_extractor_config = {'type': 'Recurrent', 'name': 'GRU', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
REC_gru_feature_dim = 128

## For GRUCell extractor
REC_gru_cell_extractor_config = {'type': 'Recurrent', 'name': 'GRUCell', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
REC_gru_cell_feature_dim = 128


###############################################################
############ Convolutional Extractor configuraiton ############
###############################################################

## For SimpleMLP extractor
MLP_mlp_extractor_config = {'type': 'Convolutional', 'name': 'MLP', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
MLP_mlp_feature_dim = 128


###########################################################
############ Attention Extractor configuraiton ############
###########################################################

## For SimpleMLP extractor
MLP_mlp_extractor_config = {'type': 'Attention', 'name': 'MLP', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
MLP_mlp_feature_dim = 128


#######################################################
############ Graph Extractor configuraiton ############
#######################################################

## For SimpleMLP extractor
MLP_mlp_extractor_config = {'type': 'Graph', 'name': 'MLP', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
MLP_mlp_feature_dim = 128


########################################################
############ Custom Extractor configuraiton ############
########################################################

## For SimpleMLP extractor
Custom_simple_mlp_extractor_config = {'type': 'Custom', 'name': 'SimpleMLP', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                                      'network_architecture': [256, 256], 'activation_function': 'relu'}
Custom_simple_mlp_feature_dim = 128

## For SimpleInception extractor
Custom_simple_inception_extractor_config = {'type': 'Custom', 'name': 'SimpleInception', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                                          'network_architecture': [256, [128, 128], 256], 'activation_function': 'relu'}
Custom_simple_inception_feature_dim = 128

## For Residual extractor
Custom_res_extractor_config = {'type': 'Custom', 'name': 'Residual', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Custom_res_feature_dim = 128

## For AE extractor
Custom_ae_extractor_config = {'type': 'Custom', 'name': 'AE', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Custom_ae_feature_dim = 128

## For UNet extractor
Custom_u_net_extractor_config = {'type': 'Custom', 'name': 'UNet', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Custom_u_net_feature_dim = 128

## For LSTM extractor
Custom_lstm_extractor_config = {'type': 'Custom', 'name': 'LSTM', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Custom_lstm_feature_dim = 128

## For CNN1D extractor
Custom_cnn1d_extractor_config = {'type': 'Custom', 'name': 'CNN1D', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Custom_cnn1d_feature_dim = 128

## For BiLSTM extractor
Custom_bi_lstm_extractor_config = {'type': 'Custom', 'name': 'BiLSTM', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Custom_bi_lstm_feature_dim = 128

## For Attention extractor
Custom_attention_extractor_config = {'type': 'Custom', 'name': 'Attention', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Custom_attention_feature_dim = 128

## For TransductiveGNN extractor
Custom_transductive_gnn_extractor_config = {'type': 'Custom', 'name': 'TransductiveGNN', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Custom_transductive_gnn_feature_dim = 128

## For InductiveGNN extractor
Custom_inductive_gnn_extractor_config = {'type': 'Custom', 'name': 'InductiveGNN', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Custom_inductive_gnn_feature_dim = 128

## For Transformer extractor
Custom_transformer_extractor_config = {'type': 'Custom', 'name': 'Transformer', 'initializer': 'glorot_normal', 'regularizer': 'l1', 'l1': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Custom_transformer_feature_dim = 128