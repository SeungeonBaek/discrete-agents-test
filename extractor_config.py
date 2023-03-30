#####################################################
############ MLP Extractor configuraiton ############
#####################################################

## For Flatten extractor
MLP_Flatten_extractor_config = {'type': 'MLP', 'name': 'Flatten', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005}
MLP_Flatten_feature_dim = 256

## For MLP extractor
MLP_MLP_extractor_config = {'type': 'MLP', 'name': 'MLP', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
MLP_MLP_feature_dim = 128

## For AutoEncoder1D extractor
MLP_AE1d_extractor_config = {'type': 'MLP', 'name': 'AutoEncoder1D', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [[256, 128], 64, [128, 256]], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
MLP_AE1d_feature_dim = 128

## For Inception1D extractor
MLP_Inception1d_extractor_config = {'type': 'MLP', 'name': 'Inception1D', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, [128, 64, 32], 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
MLP_Inception1d_feature_dim = 128

## For UNet1D extractor
MLP_UNet1d_extractor_config = {'type': 'MLP', 'name': 'UNet1D', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [[[128, 128], [64,64], [32,32]], 16, [[16, 32], [32, 64], [64,128]]],
                        'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
MLP_UNet1d_feature_dim = 128

###########################################################
############ Recurrent Extractor configuraiton ############
###########################################################

## For LSTM extractor
REC_RNN_extractor_config = {'type': 'Recurrent', 'name': 'RNN', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
REC_RNN_feature_dim = 128

## For LSTM extractor
REC_LSTM_extractor_config = {'type': 'Recurrent', 'name': 'LSTM', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
REC_LSTM_feature_dim = 128

## For GRU extractor
REC_GRU_extractor_config = {'type': 'Recurrent', 'name': 'GRU', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
REC_GRU_feature_dim = 128

## For CNN1D extractor
REC_CNN1d_extractor_config = {'type': 'Recurrent', 'name': 'CNN1D', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
REC_CNN1d_feature_dim = 128

###############################################################
############ Convolutional Extractor configuraiton ############
###############################################################

## For CNN2D extractor
Convolutional_CNN2d_extractor_config = {'type': 'Convolutional', 'name': 'CNN2D', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Convolutional_CNN2d_feature_dim = 128

## For AutoEncoder2D extractor
Convolutional_AE2d_extractor_config = {'type': 'Convolutional', 'name': 'AutoEncoder2D', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Convolutional_AE2d_feature_dim = 128

## For Inception2d extractor
Convolutional_Inception2d_extractor_config = {'type': 'Convolutional', 'name': 'Inception2d', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Convolutional_Inception2d_feature_dim = 128

## For UNet2D extractor
Convolutional_UNet2d_extractor_config = {'type': 'Convolutional', 'name': 'UNet2D', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Convolutional_UNet2d_feature_dim = 128

###########################################################
############ Attention Extractor configuraiton ############
###########################################################

## For Attention extractor
Attention_attention_extractor_config = {'type': 'Attention', 'name': 'Attention', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Attention_attention_feature_dim = 128

## For MultiHeadAttention extractor
Attention_multi_head_attention_extractor_config = {'type': 'Attention', 'name': 'MultiHeadAttention', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Attention_multi_head_attention_feature_dim = 128

## For Transformer extractor
Attention_transformer_extractor_config = {'type': 'Attention', 'name': 'Transformer', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Attention_transformer_feature_dim = 128

#######################################################
############ Graph Extractor configuraiton ############
#######################################################

## For GCN extractor
Graph_GCN_extractor_config = {'type': 'Graph', 'name': 'GCN', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Graph_GCN_feature_dim = 128

## For GAT extractor
Graph_GAT_extractor_config = {'type': 'Graph', 'name': 'GAT', 'initializer': 'glorot_normal', 'regularizer': 'l2', 'l2': 0.0005,
                        'network_architecture': [256, 256], 'use_norm': True, 'norm_type': 'layer_norm', 'act_fn': 'relu'}
Graph_GAT_feature_dim = 128

########################################################
############ Custom Extractor configuraiton ############
########################################################

## For SimpleMLP extractor
Custom_simple_mlp_extractor_config = {'type': 'Custom', 'name': 'SimpleMLP'}; Custom_simple_mlp_feature_dim = 128

## For SimpleInception extractor
Custom_simple_inception_extractor_config = {'type': 'Custom', 'name': 'SimpleInception'}; Custom_simple_inception_feature_dim = 128

## For Residual extractor
Custom_res_extractor_config = {'type': 'Custom', 'name': 'Residual'}; Custom_res_feature_dim = 128

## For AE extractor
Custom_ae_extractor_config = {'type': 'Custom', 'name': 'AE'}; Custom_ae_feature_dim = 128

## For UNet extractor
Custom_u_net_extractor_config = {'type': 'Custom', 'name': 'UNet'}; Custom_u_net_feature_dim = 128

## For SimpleInception extractor
Custom_simple_gru_extractor_config = {'type': 'Custom', 'name': 'SimpleGRU'}; Custom_simple_gru_feature_dim = 128

## For LSTM extractor
Custom_lstm_extractor_config = {'type': 'Custom', 'name': 'LSTM'}; Custom_lstm_feature_dim = 128

## For CNN1D extractor
Custom_cnn1d_extractor_config = {'type': 'Custom', 'name': 'CNN1D'}; Custom_cnn1d_feature_dim = 128

## For BiLSTM extractor
Custom_bi_lstm_extractor_config = {'type': 'Custom', 'name': 'BiLSTM'}; Custom_bi_lstm_feature_dim = 128

## For Attention extractor
Custom_attention_extractor_config = {'type': 'Custom', 'name': 'Attention'}; Custom_attention_feature_dim = 128

## For TransductiveGNN extractor
Custom_transductive_gnn_extractor_config = {'type': 'Custom', 'name': 'TransductiveGNN'}; Custom_transductive_gnn_feature_dim = 128

## For InductiveGNN extractor
Custom_inductive_gnn_extractor_config = {'type': 'Custom', 'name': 'InductiveGNN'}; Custom_inductive_gnn_feature_dim = 128

## For Transformer extractor
Custom_transformer_extractor_config = {'type': 'Custom', 'name': 'Transformer'}; Custom_transformer_feature_dim = 128