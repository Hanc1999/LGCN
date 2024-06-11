## author @Wenhui Yu  2021.01.24
## split train data into batches and train the model

import importlib

from model_MF import model_MF
from model_NCF import model_NCF
from model_GCMC import model_GCMC
from model_NGCF import model_NGCF
from model_SCF import model_SCF
from model_CGMC import model_CGMC
from model_LightGCN import model_LightGCN
from model_LCFN import model_LCFN

# LGCN
import model_LGCN
importlib.reload(model_LGCN)
from model_LGCN import model_LGCN

# LGCN tri-partite version
import model_LGCN_tri
importlib.reload(model_LGCN_tri)
from model_LGCN_tri import model_LGCN_tri

import model_LightGCN_tri
importlib.reload(model_LightGCN_tri)
from model_LightGCN_tri import model_LightGCN_tri

from model_SGNN import model_SGNN

import test_model
importlib.reload(test_model)
from test_model import test_model, test_model_train, test_model_store

import print_save
importlib.reload(print_save)

from print_save import print_value, save_value
import tensorflow as tf
import os
import numpy as np
import random as rd
import pandas as pd
import time
from tqdm import tqdm

def train_model(para, data, path_excel, results_save_path=''):
    ## data and hyperparameters
    [train_data, train_data_interaction, user_num, item_num, persona_num, test_data, pre_train_feature, hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix, _] = data
    [_, _, MODEL, LR, LAMDA, LAYER, EMB_DIM, BATCH_SIZE, TEST_USER_BATCH, N_EPOCH, IF_PRETRAIN, _, TOP_K] = para[0:13]
    if MODEL == 'LGCN' or MODEL == 'LGCN_tri': [_, _, _, KEEP_PORB, SAMPLE_RATE, GRAPH_CONV, PREDICTION, LOSS_FUNCTION, GENERALIZATION, OPTIMIZATION, IF_TRASFORMATION, ACTIVATION, POOLING] = para[13:]
    if MODEL == 'SGNN': [_, PROP_EMB, _] = para[13:]
    para_test = [train_data, test_data, user_num, item_num, TOP_K, TEST_USER_BATCH]
    ## Define the model
    if MODEL == 'MF': model = model_MF(n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA)
    if MODEL == 'NCF': model = model_NCF(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN)
    if MODEL == 'GCMC': model = model_GCMC(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'NGCF': model = model_NGCF(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'SCF': model = model_SCF(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'CGMC': model = model_CGMC(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'LightGCN': model = model_LightGCN(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)
    if MODEL == 'LCFN': model = model_LCFN(layer=LAYER, n_users=user_num, n_items=item_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, graph_embeddings=hypergraph_embeddings)
    if MODEL == 'LGCN': model = model_LGCN(n_users=user_num, n_items=item_num, lr=LR, lamda=LAMDA, emb_dim=EMB_DIM, layer=LAYER, pre_train_latent_factor=pre_train_feature, graph_embeddings=graph_embeddings, graph_conv = GRAPH_CONV, prediction = PREDICTION, loss_function=LOSS_FUNCTION, generalization = GENERALIZATION, optimization=OPTIMIZATION, if_pretrain=IF_PRETRAIN, if_transformation=IF_TRASFORMATION, activation=ACTIVATION, pooling=POOLING)
    if MODEL == 'SGNN': model = model_SGNN(n_users=user_num, n_items=item_num, lr=LR, lamda=LAMDA, emb_dim=EMB_DIM, layer=LAYER, pre_train_latent_factor=pre_train_feature, propagation_embeddings=propagation_embeddings, if_pretrain=IF_PRETRAIN, prop_emb=PROP_EMB)
    if MODEL == 'LGCN_tri': model = model_LGCN_tri(n_users=user_num, n_items=item_num, n_personas=persona_num, lr=LR, lamda=LAMDA, emb_dim=EMB_DIM, layer=LAYER, pre_train_latent_factor=pre_train_feature, graph_embeddings=graph_embeddings, graph_conv = GRAPH_CONV, prediction = PREDICTION, loss_function=LOSS_FUNCTION, generalization = GENERALIZATION, optimization=OPTIMIZATION, if_pretrain=IF_PRETRAIN, if_transformation=IF_TRASFORMATION, activation=ACTIVATION, pooling=POOLING)
    if MODEL == 'LightGCN_tri': model = model_LightGCN_tri(layer=LAYER, n_users=user_num, n_items=item_num, n_personas=persona_num, emb_dim=EMB_DIM, lr=LR, lamda=LAMDA, pre_train_latent_factor=pre_train_feature, if_pretrain=IF_PRETRAIN, sparse_graph=sparse_propagation_matrix)

    # debug,
    # return model

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())

    ## Split the training samples into batches
    batches = list(range(0, len(train_data_interaction), BATCH_SIZE))
    batches.append(len(train_data_interaction))
    ## Training iteratively
    F1_max = 0
    F1_df = pd.DataFrame(columns=TOP_K)
    F1_df_train = pd.DataFrame(columns=TOP_K) # to log the training loss's changes
    NDCG_df = pd.DataFrame(columns=TOP_K)
    t1 = time.perf_counter()

    # training loops
    with tqdm(total=N_EPOCH) as pbar:
        for epoch in range(N_EPOCH):
            for batch_num in range(len(batches) - 1):
                train_batch_data = []
                for sample in range(batches[batch_num], batches[batch_num + 1]):
                    (user, pos_item) = train_data_interaction[sample]
                    sample_num = 0
                    while sample_num < (SAMPLE_RATE if (MODEL == 'LGCN' or MODEL == 'LGCN_tri') else 1):
                        neg_item = int(rd.uniform(0, item_num)) # sample random exclusive items as the negative
                        if not (neg_item in train_data[user]):
                            sample_num += 1
                            train_batch_data.append([user, pos_item, neg_item])
                train_batch_data = np.array(train_batch_data)
                _, loss = sess.run([model.updates, model.loss], feed_dict={model.users: train_batch_data[:, 0], model.pos_items: train_batch_data[:, 1], model.neg_items: train_batch_data[:, 2], model.keep_prob: KEEP_PORB if MODEL in ['LGCN', 'LGCN_tri'] else 1})
            ## test the model each epoch
            F1, NDCG = test_model(sess, model, para_test)
            F1_max = max(F1_max, F1[0])
            # F1_train, NDCG_train = test_model_train(sess, model, para_test)
            ## print performance
            # print_value([epoch + 1, loss, F1_max, F1, NDCG])
            # if epoch % 10 == 0: print('%.5f' % (F1_max), end = ' ', flush = True)
            pbar.set_description(f"F1_max: {F1_max :2f}")
            pbar.update(1)
            ## save performance
            F1_df.loc[epoch + 1] = F1
            NDCG_df.loc[epoch + 1] = NDCG
            # F1_df_train.loc[epoch + 1] = F1_train
            # save_value([[F1_df, 'F1'], [F1_df_train, 'F1_train'], [NDCG_df, 'NDCG']], path_excel, first_sheet=False)
            save_value([[F1_df, 'F1'], [NDCG_df, 'NDCG']], path_excel, first_sheet=False)
            if loss > 10 ** 10: break
    t2 = time.perf_counter()
    print('time cost:', (t2 - t1) / 200)
    
    if results_save_path:
        print('Saving results...')
        test_model_store(sess, model, para_test, results_save_path)
        print('Well saved.')
    
    return F1_max