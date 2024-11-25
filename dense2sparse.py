## author@ Wenhui Yu email: jianlin.ywh@alibaba-inc.com  2021.02.16
## constructing the sparse graph

import tensorflow as tf
import numpy as np

def propagation_matrix(graph, user_num, item_num, norm):
    print('Constructing the sparse graph...')
    eps = 0.1 ** 10
    user_itemNum = np.zeros(user_num)
    item_userNum = np.zeros(item_num)
    for (user, item) in graph:
        user_itemNum[user] += 1
        item_userNum[item] += 1
    val, idx = [], []
    for (user, item) in graph:
        if norm == 'left_norm':
            val += [1 / max(user_itemNum[user], eps), 1 / max(item_userNum[item], eps)]
            idx += [[user, item + user_num], [item + user_num, user]]
        if norm == 'sym_norm':
            val += [1 / (max(np.sqrt(user_itemNum[user] * item_userNum[item]), eps))] * 2
            idx += [[user, item + user_num], [item + user_num, user]]
    return tf.SparseTensor(indices=idx, values=val, dense_shape=[user_num + item_num, user_num + item_num])

def propagation_matrix_tri(graph, user_num, item_num, persona_num, norm):
    print('Constructing the tripartite sparse graph...')
    [uidx2tidx_graph, uidx2pidx_graph, tidx2pidx_graph] = graph # parse the 3 subgraphs
    
    eps = 0.1 ** 10
    user_degreeNum = np.zeros(user_num)
    item_degreeNum = np.zeros(item_num)
    persona_degreeNum = np.zeros(persona_num)
    
    for (user, item) in uidx2tidx_graph:
        user_degreeNum[user] += 1
        item_degreeNum[item] += 1

    for (user, persona) in uidx2pidx_graph:
        user_degreeNum[user] += 1
        persona_degreeNum[persona] += 1
    
    for (item, persona) in tidx2pidx_graph:
        item_degreeNum[item] += 1
        persona_degreeNum[persona] += 1

    if norm == 'sym_norm':
        val, idx = [], []
        for (user, item) in uidx2tidx_graph:
            val += [1 / (max(np.sqrt(user_degreeNum[user] * item_degreeNum[item]), eps))] * 2
            idx += [[user, item + user_num], [item + user_num, user]]
        for (user, persona) in uidx2pidx_graph:
            val += [1 / (max(np.sqrt(user_degreeNum[user] * persona_degreeNum[persona]), eps))] * 2
            idx += [[user, persona + user_num + item_num], [persona + user_num + item_num, user]]
        for (item, persona) in tidx2pidx_graph:
            val += [1 / (max(np.sqrt(item_degreeNum[item] * persona_degreeNum[persona]), eps))] * 2
            idx += [[item + user_num, persona + user_num + item_num], [persona + user_num + item_num, item + user_num]]
        return tf.SparseTensor(indices=idx, values=val, dense_shape=[user_num + item_num + persona_num, user_num + item_num + persona_num])
    
    else: assert False, f'Not supported: {norm}'

def propagation_matrix_rgcn(graph, user_num, item_num, persona_num, norm):
    print('Constructing the rgcn sparse graph...')
    [uidx2tidx_graph, uidx2pidx_graph, tidx2pidx_graph] = graph # parse the 3 subgraphs
    if norm != 'sym_norm': assert False, f'Not supported: {norm}'
    all_num = user_num + item_num + persona_num

    eps = 0.1 ** 10
    
    # for A_{uv}
    user_degreeNum_uv = np.zeros(user_num)
    item_degreeNum_uv = np.zeros(item_num)
    for (user, item) in uidx2tidx_graph:
        user_degreeNum_uv[user] += 1
        item_degreeNum_uv[item] += 1
    val_uv, idx_uv = [], []
    for (user, item) in uidx2tidx_graph:
        val_uv += [4/5 / (max(np.sqrt(user_degreeNum_uv[user] * item_degreeNum_uv[item]), eps))]
        val_uv += [4/5 / (max(np.sqrt(user_degreeNum_uv[user] * item_degreeNum_uv[item]), eps))]
        idx_uv += [[user, item + user_num], [item + user_num, user]]
    val_uv = tf.constant(val_uv, dtype=tf.float32)
    A_hat_uv = tf.SparseTensor(indices=idx_uv, values=val_uv, dense_shape=[all_num, all_num])
    # signal_diag_uv = tf.linalg.diag(tf.constant([1/2]*all_num, dtype=tf.float32)) # dense diag matrix
    # signal_diag_uv = tf.SparseTensor(indices=[[i,i] for i in range(all_num)], values=[1/2]*all_num, dense_shape=[all_num, all_num])
    # TODO: add signal control for signal_diag matrices
    
    # for A_{ur}
    user_degreeNum_ur = np.zeros(user_num)
    persona_degreeNum_ur = np.zeros(persona_num)
    for (user, persona) in uidx2pidx_graph:
        user_degreeNum_ur[user] += 1
        persona_degreeNum_ur[persona] += 1
    val_ur, idx_ur = [], []
    for (user, persona) in uidx2pidx_graph:
        val_ur += [1/5 / (max(np.sqrt(user_degreeNum_ur[user] * persona_degreeNum_ur[persona]), eps))]
        val_ur += [1/2 / (max(np.sqrt(user_degreeNum_ur[user] * persona_degreeNum_ur[persona]), eps))]
        idx_ur += [[user, persona + user_num + item_num], [persona + user_num + item_num, user]]
    val_ur = tf.constant(val_ur, dtype=tf.float32)
    A_hat_ur = tf.SparseTensor(indices=idx_ur, values=val_ur, dense_shape=[all_num, all_num])
    # signal_diag_ur = tf.linalg.diag(tf.constant([1/2]*all_num, dtype=tf.float32))
    # signal_diag_ur = tf.SparseTensor(indices=[[i,i] for i in range(all_num)], values=[1/2]*all_num, dense_shape=[all_num, all_num])

    # for A_{vr}
    item_degreeNum_vr = np.zeros(item_num)
    persona_degreeNum_vr = np.zeros(persona_num)
    for (item, persona) in tidx2pidx_graph:
        item_degreeNum_vr[item] += 1
        persona_degreeNum_vr[persona] += 1
    val_vr, idx_vr = [], []
    for (item, persona) in tidx2pidx_graph:
        val_vr += [1/5 / (max(np.sqrt(item_degreeNum_vr[item] * persona_degreeNum_vr[persona]), eps))]
        val_vr += [1/2 / (max(np.sqrt(item_degreeNum_vr[item] * persona_degreeNum_vr[persona]), eps))]
        idx_vr += [[item, persona + user_num + item_num], [persona + user_num + item_num, item]]
    val_vr = tf.constant(val_vr, dtype=tf.float32)
    A_hat_vr = tf.SparseTensor(indices=idx_vr, values=val_vr, dense_shape=[all_num, all_num])
    # signal_diag_vr = tf.linalg.diag(tf.constant([1/2]*all_num, dtype=tf.float32))
    # signal_diag_vr = tf.SparseTensor(indices=[[i,i] for i in range(all_num)], values=[1/2]*all_num, dense_shape=[all_num, all_num])

    # A_hat_rgcn = tf.sparse.add(tf.sparse.add(tf.sparse.sparse_dense_matmul(A_hat_uv, signal_diag_uv), tf.sparse.sparse_dense_matmul(A_hat_ur, signal_diag_ur)), tf.sparse.sparse_dense_matmul(A_hat_vr, signal_diag_vr))
    A_hat_rgcn = tf.sparse.add(tf.sparse.add(A_hat_uv, A_hat_ur), A_hat_vr)
    print(A_hat_rgcn)
    return A_hat_rgcn
