## author@Wenhui Yu  2021.02.16
## read train/test/validation data
## transform data into wanted structures
## return user and item number, and padding train data
## read (hyper-) graph embeddings, propoagation embeddings, and pre-trained embeddings
## construct sparse graph

import json
import numpy as np
import random as rd
import importlib

import dense2sparse
importlib.reload(dense2sparse)
from dense2sparse import propagation_matrix, propagation_matrix_tri

def read_data(path):
    with open(path) as f:
        line = f.readline()
        data = json.loads(line)
    f.close()
    user_num = len(data)
    item_num = 0
    interactions = []
    for user in range(user_num):
        for item in data[user]:
            interactions.append((user, item))
            item_num = max(item, item_num)
    item_num += 1
    rd.shuffle(interactions)
    return(data, interactions, user_num, item_num)

# def read_data_tri(path_u2t, path_t2p, path_u2p,):
#     with open(path_u2t, 'r') as f:
#         tri_graph_uidx2tidx_train = json.load(f)
#     # trans key to int
#     tri_graph_uidx2tidx_train = {int(k):v for k,v in tri_graph_uidx2tidx_train.items()}
#     # make data and interactions
#     user_num = len(tri_graph_uidx2tidx_train)
#     data = [tri_graph_uidx2tidx_train[uidx] for uidx in range(user_num)] # [[tidx,]]
#     interactions = []
#     for user in range(user_num): # user: uidx
#         for item in data[user]: # item: tidx
#             interactions.append((user, item))
#     rd.shuffle(interactions) # uncontrolled shuffle
#     # item number
#     with open(path_t2p, 'r') as f:
#         tri_graph_tidx2pidx = json.load(f) # just to check the length
#     item_num = len(tri_graph_tidx2pidx)

#     return(data, interactions, user_num, item_num)

def read_data_tri(path_u2t, path_t2p, path_u2p,):
    # read files
    with open(path_u2t, 'r') as f:
        tri_graph_uidx2tidx_train = json.load(f)
    with open(path_t2p, 'r') as f:
        tri_graph_tidx2pidx = json.load(f)
    with open(path_u2p, 'r') as f:
        tri_graph_uidx2pidx = json.load(f)

    # trans key to int
    tri_graph_uidx2tidx_train = {int(k):v for k,v in tri_graph_uidx2tidx_train.items()}
    tri_graph_tidx2pidx = {int(k):v for k,v in tri_graph_tidx2pidx.items()}
    tri_graph_uidx2pidx = {int(k):v for k,v in tri_graph_uidx2pidx.items()}

    # print(len(tri_graph_uidx2tidx_train), len(tri_graph_tidx2pidx), len(tri_graph_uidx2pidx)) # debug

    # make data
    user_num = len(tri_graph_uidx2tidx_train)
    item_num = len(tri_graph_tidx2pidx)

    u2t_data = [tri_graph_uidx2tidx_train[uidx] for uidx in range(user_num)] # [[tidx,]]
    u2p_data = [tri_graph_uidx2pidx.get(uidx, []) for uidx in range(user_num)] # [[pidx,]]
    # notice: u2p is not available for some users
    t2p_data = [tri_graph_tidx2pidx[tidx] for tidx in range(item_num)] # [[pidx,]]

    # make interactions
    interactions_u2t = []
    interactions_u2p = []
    interactions_t2p = []
    for user in range(user_num): # user: uidx
        for item in u2t_data[user]: # item: tidx
            interactions_u2t.append((user, item))
        for persona in u2p_data[user]:
            interactions_u2p.append((user, persona))
    for item in range(item_num):
        for persona in t2p_data[item]:
            interactions_t2p.append((item, persona))
    
    # shuffle for randomness
    rd.shuffle(interactions_u2t) # [TODO] uncontrolled shuffle
    rd.shuffle(interactions_u2p)
    rd.shuffle(interactions_t2p)
    
    return(u2t_data, interactions_u2t, interactions_u2p, interactions_t2p, user_num, item_num)

def read_bases(path, fre_u, fre_v):
    with open(path) as f:
        line = f.readline()
        bases = json.loads(line)
    f.close()
    [feat_u, feat_v] = bases
    feat_u = np.array(feat_u)[:, 0: fre_u].astype(np.float32)
    feat_v = np.array(feat_v)[:, 0: fre_v].astype(np.float32)
    return [feat_u, feat_v]

def read_bases1(path, fre, _if_norm = False):
    with open(path) as f:
        line = f.readline()
        bases = json.loads(line)
    f.close()
    if _if_norm:
        for i in range(len(bases)):
            bases[i] = bases[i]/np.sqrt(np.dot(bases[i], bases[i]))
    return np.array(bases)[:, 0: fre].astype(np.float32)

def read_all_data(all_para):
    [_, DATASET, MODEL, _, _, _, EMB_DIM, _, _, _, IF_PRETRAIN, TEST_VALIDATION, _, FREQUENCY_USER, FREQUENCY_ITEM, FREQUENCY, _, _, GRAPH_CONV, _, _, _, _, _, _, _, PROP_DIM, PROP_EMB, IF_NORM] = all_para
    [hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix] = [0, 0, 0, 0]

    ## Paths of data
    DIR = 'dataset/' + DATASET + '/'
    train_path = DIR + 'train_data.json'
    test_path = DIR + 'test_data.json'
    validation_path = DIR + 'validation_data.json'
    hypergraph_embeddings_path = DIR + 'hypergraph_embeddings.json'                     # hypergraph embeddings
    graph_embeddings_1d_path = DIR + 'graph_embeddings_1d.json'                         # 1d graph embeddings
    graph_embeddings_2d_path = DIR + 'graph_embeddings_2d.json'                         # 2d graph embeddings
    pre_train_feature_path = DIR + 'pre_train_feature' + str(EMB_DIM) + '.json'         # pretrained latent factors
    if MODEL == 'SGNN': propagation_embeddings_path = DIR + 'pre_train_feature' + str(PROP_DIM) + '.json'   # pretrained latent factors

    ## Load data
    ## load training data
    print('Reading data...')
    [train_data, train_data_interaction, user_num, item_num] = read_data(train_path)
    ## load test data
    test_vali_path = validation_path if TEST_VALIDATION == 'Validation' else test_path
    test_data = read_data(test_vali_path)[0]
    ## load pre-trained embeddings for all deep models
    if IF_PRETRAIN:
        try: pre_train_feature = read_bases(pre_train_feature_path, EMB_DIM, EMB_DIM)
        except:
            print('There is no pre-trained embeddings found!!')
            pre_train_feature = [0, 0]
            IF_PRETRAIN = False
    else:
        pre_train_feature = [0, 0]

    ## load pre-trained transform bases for LCFN and SGNN
    if MODEL == 'LCFN': hypergraph_embeddings = read_bases(hypergraph_embeddings_path, FREQUENCY_USER, FREQUENCY_ITEM)
    if MODEL == 'LGCN':
        if GRAPH_CONV == '1D': graph_embeddings = read_bases1(graph_embeddings_1d_path, FREQUENCY)
        if GRAPH_CONV == '2D_graph': graph_embeddings = read_bases(graph_embeddings_2d_path, FREQUENCY_USER, FREQUENCY_ITEM)
        if GRAPH_CONV == '2D_hyper_graph': graph_embeddings = read_bases(hypergraph_embeddings_path, FREQUENCY_USER, FREQUENCY_ITEM)
    if MODEL == 'SGNN':
        if PROP_EMB == 'RM': propagation_embeddings = read_bases(propagation_embeddings_path, PROP_DIM, PROP_DIM)
        if PROP_EMB == 'SF': propagation_embeddings = read_bases1(graph_embeddings_1d_path, PROP_DIM, IF_NORM)
        if PROP_EMB == 'PE': propagation_embeddings = 0

    ## convert dense graph to sparse graph
    if MODEL in ['GCMC', 'SCF', 'CGMC']: sparse_propagation_matrix = propagation_matrix(train_data_interaction, user_num, item_num, 'left_norm')
    elif MODEL in ['NGCF', 'LightGCN']: sparse_propagation_matrix = propagation_matrix(train_data_interaction, user_num, item_num, 'sym_norm')

    print('Data all read successfully!')
    persona_num = 0
    return train_data, train_data_interaction, user_num, item_num, persona_num, test_data, pre_train_feature, hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix, IF_PRETRAIN

def read_all_data_tri(all_para, approximate=False):
    # approximate=True will read the approximated ver of tri_graph_uidx2pidx
    [_, DATASET, MODEL, _, _, _, EMB_DIM, _, _, _, IF_PRETRAIN, TEST_VALIDATION, _, FREQUENCY_USER, FREQUENCY_ITEM, FREQUENCY, _, _, GRAPH_CONV, _, _, _, _, _, _, _, PROP_DIM, PROP_EMB, IF_NORM] = all_para
    [hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix] = [0, 0, 0, 0]

    ## Paths of data
    DIR = 'dataset/' + DATASET + '/'
    # hypergraph_embeddings_path = DIR + 'hypergraph_embeddings.json'                   # hypergraph embeddings

    ## Load data
    ## load training data
    print('Reading data...')
    path_u2t = DIR + 'tri_graph_uidx2tidx_train.json'
    path_t2p = DIR + 'tri_graph_tidx2pidx_lifestyle.json'
    if approximate:
        # path_u2p = DIR + 'tri_graph_uidx2pidx_approach.json'
        path_u2p = DIR + 'tri_graph_uidx2pidx_app_e_0.001.json'
    else:
        path_u2p = DIR + 'tri_graph_uidx2pidx_lifestyle.json'
    # [train_data, train_data_interaction, user_num, item_num] = read_data_tri(path_u2t, path_t2p)
    [train_data, train_data_interaction, interactions_u2p, interactions_t2p, user_num, item_num] = read_data_tri(path_u2t, path_t2p, path_u2p)
    
    ## load test data
    test_vali_path = DIR + 'tri_graph_uidx2tidx_valid.json' if TEST_VALIDATION == 'Validation' else DIR + 'tri_graph_uidx2tidx_test.json'
    test_data = read_data_tri(test_vali_path, path_t2p, path_u2p)[0]
    
    if DATASET == 'MBA':
        persona_num = 20 # 20, for MBA
    elif DATASET == 'Instacart':
        persona_num = 51 # 51, fixed for Instacart
    elif DATASET == 'Instacart_full':
        persona_num = 51
    
    # graph_embeddings_2d_path = DIR + 'graph_embeddings_2d.json'                         # 2d graph embeddings

    if MODEL in ['LGCN', 'LGCN_tri']:
        if GRAPH_CONV == '1D':
            if not approximate:
                # normal case
                graph_embeddings_1d_path = DIR + 'graph_embeddings_1d_tri.json' if MODEL == 'LGCN_tri' else  DIR + 'graph_embeddings_1d.json'   # 1d graph embeddings
            else:
                # approximated case
                graph_embeddings_1d_path = DIR + 'graph_embeddings_1d_tri_approach.json' if MODEL == 'LGCN_tri' else  DIR + 'graph_embeddings_1d.json'   # 1d graph embeddings

            print(f'Reading graph_embeddings_1d from path: {graph_embeddings_1d_path}')
            ## load pre-trained transform bases for LCFN and SGNN
            graph_embeddings = read_bases1(graph_embeddings_1d_path, FREQUENCY) # same as LGCN
        else: assert False, f'Not supported: {MODEL}, {GRAPH_CONV}'
    elif MODEL in ['LightGCN', 'LightGCN_tri', 'NGCF', 'GCMC']:
        if MODEL == 'LightGCN_tri':
            graph_tri = [train_data_interaction, interactions_u2p, interactions_t2p]
            sparse_propagation_matrix = propagation_matrix_tri(graph_tri, user_num, item_num, persona_num, 'sym_norm')
        elif MODEL == 'LightGCN':
            sparse_propagation_matrix = propagation_matrix(train_data_interaction, user_num, item_num, 'sym_norm')
        elif MODEL == 'NGCF':
            sparse_propagation_matrix = propagation_matrix(train_data_interaction, user_num, item_num, 'sym_norm')
        elif MODEL == 'GCMC':
            sparse_propagation_matrix = propagation_matrix(train_data_interaction, user_num, item_num, 'left_norm')
    
    pre_train_feature_path = DIR + 'pre_train_feature' + str(EMB_DIM) + '.json'         # pretrained latent factors
    ## load pre-trained embeddings for all deep models
    if IF_PRETRAIN:
        try: pre_train_feature = read_bases(pre_train_feature_path, EMB_DIM, EMB_DIM)
        except:
            print('There is no pre-trained embeddings found!!')
            pre_train_feature = [0, 0]
            IF_PRETRAIN = False
    else:
        pre_train_feature = [0, 0]

    print('Data all read successfully!')
    return train_data, train_data_interaction, user_num, item_num, persona_num, test_data, pre_train_feature, hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix, IF_PRETRAIN
    # 0:train_data, 1:train_data_interaction, 2:user_num, 3:item_num, 4:persona_num, 5:test_data,
    # 6:pre_train_feature, 7:hypergraph_embeddings, 8:graph_embeddings, 9:propagation_embeddings,
    # 10:sparse_propagation_matrix, 11:IF_PRETRAIN


def read_all_data_lightGCN(all_para):
    [_, DATASET, MODEL, _, _, _, EMB_DIM, _, _, _, IF_PRETRAIN, TEST_VALIDATION, _, FREQUENCY_USER, FREQUENCY_ITEM, FREQUENCY, _, _, GRAPH_CONV, _, _, _, _, _, _, _, PROP_DIM, PROP_EMB, IF_NORM] = all_para
    [hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix] = [0, 0, 0, 0]

    ## Paths of data
    DIR = 'dataset/' + DATASET + '/'
    # train_path = DIR + 'train_data.json'
    # test_path = DIR + 'test_data.json'
    pathu2t = DIR + 'tri_graph_uidx2tidx_train.json'
    patht2p = DIR + 'tri_graph_tidx2pidx.json'

    validation_path = DIR + 'validation_data.json'
    hypergraph_embeddings_path = DIR + 'hypergraph_embeddings.json'                     # hypergraph embeddings
    graph_embeddings_1d_path = DIR + 'graph_embeddings_1d.json'                         # 1d graph embeddings
    graph_embeddings_2d_path = DIR + 'graph_embeddings_2d.json'                         # 2d graph embeddings
    pre_train_feature_path = DIR + 'pre_train_feature' + str(EMB_DIM) + '.json'         # pretrained latent factors
    if MODEL == 'SGNN': propagation_embeddings_path = DIR + 'pre_train_feature' + str(PROP_DIM) + '.json'   # pretrained latent factors

    ## Load data
    ## load training data
    print('Reading data...')
    [train_data, train_data_interaction, user_num, item_num] = read_data_tri(pathu2t, patht2p)

    ## load test data
    test_vali_path = DIR + 'tri_graph_uidx2tidx_valid.json' if TEST_VALIDATION == 'Validation' else DIR + 'tri_graph_uidx2tidx_test.json'
    test_data = read_data_tri(test_vali_path, patht2p)[0]

    ## load pre-trained embeddings for all deep models
    if IF_PRETRAIN:
        try: pre_train_feature = read_bases(pre_train_feature_path, EMB_DIM, EMB_DIM)
        except:
            print('There is no pre-trained embeddings found!!')
            pre_train_feature = [0, 0]
            IF_PRETRAIN = False
    else:
        pre_train_feature = [0, 0]

    ## load pre-trained transform bases for LCFN and SGNN
    # if MODEL == 'LCFN': hypergraph_embeddings = read_bases(hypergraph_embeddings_path, FREQUENCY_USER, FREQUENCY_ITEM)
    # if MODEL == 'LGCN':
    #     if GRAPH_CONV == '1D': graph_embeddings = read_bases1(graph_embeddings_1d_path, FREQUENCY)
    #     if GRAPH_CONV == '2D_graph': graph_embeddings = read_bases(graph_embeddings_2d_path, FREQUENCY_USER, FREQUENCY_ITEM)
    #     if GRAPH_CONV == '2D_hyper_graph': graph_embeddings = read_bases(hypergraph_embeddings_path, FREQUENCY_USER, FREQUENCY_ITEM)
    # if MODEL == 'SGNN':
    #     if PROP_EMB == 'RM': propagation_embeddings = read_bases(propagation_embeddings_path, PROP_DIM, PROP_DIM)
    #     if PROP_EMB == 'SF': propagation_embeddings = read_bases1(graph_embeddings_1d_path, PROP_DIM, IF_NORM)
    #     if PROP_EMB == 'PE': propagation_embeddings = 0

    # ## convert dense graph to sparse graph
    # if MODEL in ['GCMC', 'SCF', 'CGMC']: sparse_propagation_matrix = propagation_matrix(train_data_interaction, user_num, item_num, 'left_norm')
    if MODEL in ['LightGCN']: sparse_propagation_matrix = propagation_matrix(train_data_interaction, user_num, item_num, 'sym_norm')

    print('Data all read successfully!')
    persona_num = 0
    return train_data, train_data_interaction, user_num, item_num, persona_num, test_data, pre_train_feature, hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix, IF_PRETRAIN


def read_all_data_LCFN(all_para):
    [_, DATASET, MODEL, _, _, _, EMB_DIM, _, _, _, IF_PRETRAIN, TEST_VALIDATION, _, FREQUENCY_USER, FREQUENCY_ITEM, FREQUENCY, _, _, GRAPH_CONV, _, _, _, _, _, _, _, PROP_DIM, PROP_EMB, IF_NORM] = all_para
    [hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix] = [0, 0, 0, 0]

    ## Paths of data
    DIR = 'dataset/' + DATASET + '/'
    pathu2t = DIR + 'tri_graph_uidx2tidx_train.json'
    patht2p = DIR + 'tri_graph_tidx2pidx.json'

    # train_path = DIR + 'train_data.json'
    # test_path = DIR + 'test_data.json'
    # validation_path = DIR + 'validation_data.json'
    hypergraph_embeddings_path = DIR + 'hypergraph_embeddings.json'                     # hypergraph embeddings
    # graph_embeddings_1d_path = DIR + 'graph_embeddings_1d.json'                         # 1d graph embeddings
    # graph_embeddings_2d_path = DIR + 'graph_embeddings_2d.json'                         # 2d graph embeddings
    pre_train_feature_path = DIR + 'pre_train_feature' + str(EMB_DIM) + '.json'         # pretrained latent factors

    ## Load data
    ## load training data
    print('Reading data...')
    [train_data, train_data_interaction, user_num, item_num] = read_data_tri(pathu2t, patht2p)
    ## load test data
    test_vali_path = DIR + 'tri_graph_uidx2tidx_valid.json' if TEST_VALIDATION == 'Validation' else DIR + 'tri_graph_uidx2tidx_test.json'
    test_data = read_data_tri(test_vali_path, patht2p)[0]
    ## load pre-trained embeddings for all deep models
    if IF_PRETRAIN:
        try: pre_train_feature = read_bases(pre_train_feature_path, EMB_DIM, EMB_DIM)
        except:
            print('There is no pre-trained embeddings found!!')
            pre_train_feature = [0, 0]
            IF_PRETRAIN = False
    else:
        pre_train_feature = [0, 0]

    ## load pre-trained transform bases for LCFN and SGNN
    if MODEL == 'LCFN': hypergraph_embeddings = read_bases(hypergraph_embeddings_path, FREQUENCY_USER, FREQUENCY_ITEM)

    print('Data all read successfully!')
    persona_num = 0
    return train_data, train_data_interaction, user_num, item_num, persona_num, test_data, pre_train_feature, hypergraph_embeddings, graph_embeddings, propagation_embeddings, sparse_propagation_matrix, IF_PRETRAIN
