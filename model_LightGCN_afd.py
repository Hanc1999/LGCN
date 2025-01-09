## baseline: Light Graph Convolutional Network (LightGCN)
## Xiangnan He and Kuan Deng and Xiang Wang and Yan Li and Yong-Dong Zhang and Meng Wang. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR, 2020.

import tensorflow as tf
import numpy as np
from afd_utils import calculate_correlation

class model_LightGCN_afd(object):
    def __init__(self, layer, n_users, n_items, emb_dim, lr, lamda, pre_train_latent_factor, if_pretrain, sparse_graph, afd_alpha):
        self.model_name = 'LightGCN'
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.layer = layer
        self.lamda = lamda
        self.lr = lr
        self.alpha = afd_alpha
        [self.U, self.V] = pre_train_latent_factor
        self.if_pretrain = if_pretrain
        self.A_hat = sparse_graph
        self.layer_weight = [1/(i + 1) for i in range(self.layer + 1)]

        # placeholder definition
        # tf.compat.v1.disable_eager_execution() # to disable the eager mode

        self.users = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, shape=(None))
        self.items_in_train_data = tf.compat.v1.placeholder(tf.float32, shape=(None, None))
        self.top_k = tf.compat.v1.placeholder(tf.int32, shape=(None))

        if self.if_pretrain:
            self.user_embeddings = tf.Variable(self.U, name='user_embeddings')
            self.item_embeddings = tf.Variable(self.V, name='item_embeddings')
        else:
            self.user_embeddings = tf.Variable(
                tf.random.normal([self.n_users, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
                name='user_embeddings')
            self.item_embeddings = tf.Variable(
                tf.random.normal([self.n_items, self.emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32),
                name='item_embeddings')
        embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        all_embeddings = embeddings # the self embedding is considered by default
        embeddings_list = [] # added for afd
        for l in range(self.layer):
            embeddings = tf.sparse.sparse_dense_matmul(self.A_hat, embeddings)
            embeddings_list.append(embeddings)
            all_embeddings += embeddings * self.layer_weight[l + 1]
        self.user_all_embeddings, self.item_all_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        self.u_embeddings = tf.nn.embedding_lookup(self.user_all_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.pos_items)
        self.neg_i_embeddings = tf.nn.embedding_lookup(self.item_all_embeddings, self.neg_items)

        self.u_embeddings_reg = tf.nn.embedding_lookup(self.user_embeddings, self.users)
        self.pos_i_embeddings_reg = tf.nn.embedding_lookup(self.item_embeddings, self.pos_items)
        self.neg_i_embeddings_reg = tf.nn.embedding_lookup(self.item_embeddings, self.neg_items)
        
        afd_u, afd_i = self.afd_loss(embeddings_list)
        self.loss = self.create_bpr_loss(self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings) + \
                    self.lamda * self.regularization(self.u_embeddings_reg, self.pos_i_embeddings_reg,
                                                     self.neg_i_embeddings_reg) + \
                    afd_u + afd_i
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
        self.updates = self.opt.minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings])

        self.all_ratings = tf.matmul(self.u_embeddings, self.item_all_embeddings, transpose_a=False, transpose_b=True)
        self.all_ratings += self.items_in_train_data  ## set a very small value for the items appearing in the training set to make sure they are at the end of the sorted list
        self.top_items = tf.nn.top_k(self.all_ratings, k=self.top_k, sorted=True).indices

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        maxi = tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_sum(maxi))
        return loss

    def regularization(self, users, pos_items, neg_items):
        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        return regularizer

    def afd_loss(self, embedding_list):
        # cor_loss_u = tf.Variable(tf.zeros((1,), dtype=tf.float32))
        # cor_loss_i = tf.Variable(tf.zeros((1,), dtype=tf.float32))
        cor_loss_u = tf.zeros((1,), dtype=tf.float32)
        cor_loss_i = tf.zeros((1,), dtype=tf.float32)
        
        user_layer_correlations = []
        item_layer_correlations = []

        for i in range(1, self.layer + 1):
            user_embeddings, item_embeddings = tf.split(embeddings_list[i], [self.n_users, self.n_items], axis=0)
            user_layer_correlations.append(calculate_correlation(user_embeddings))
            item_layer_correlations.append(calculate_correlation(item_embeddings))
            
        user_layer_correlations = tf.convert_to_tensor(user_layer_correlations) #tf.convert_to_tensor() converts any value object to a tensor
        item_layer_correlations = tf.convert_to_tensor(item_layer_correlations)
        
        user_layer_correlations_coef = (1 / user_layer_correlations) / tf.reduce_sum(1 / user_layer_correlations)
        item_layer_correlations_coef = (1 / item_layer_correlations) / tf.reduce_sum(1 / item_layer_correlations)

        for i in range(1, self.layer + 1):
            cor_loss_u = cor_loss_u + user_layer_correlations_coef[i - 1] * user_layer_correlations[i - 1]
            cor_loss_i = cor_loss_i + item_layer_correlations_coef[i - 1] * item_layer_correlations[i - 1]

        return self.alpha * cor_loss_u[0], self.alpha * cor_loss_i[0]
