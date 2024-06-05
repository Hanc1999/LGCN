## hyper-parameter setting
## author@Wenhui Yu  2021.02.16
## email: jianlin.ywh@alibaba-inc.com

model = 10 # 10          # 0:MF, 1:NCF, 2:GCMC, 3:NGCF, 4:SCF, 5:CGMC, 6:LightGCN, 7:LCFN, 8:LGCN, 9:SGNN, 10:LGCN_tri, 11:LightGCN_tri
dataset = 3         # 0:Amazon, 1:Movielens, 2: MBA, 3: Instacart
pred_dim = 128      # predictive embedding dimensionality (must align with the pretraining)

## parameters about experiment setting
GPU_INDEX = "0"
DATASET = ['Amazon', 'Movielens', 'MBA', 'Instacart'][dataset]
MODEL_list = ['MF', 'NCF', 'GCMC', 'NGCF', 'SCF', 'CGMC', 'LightGCN', 'LCFN', 'LGCN', 'SGNN', 'LGCN_tri', 'LightGCN_tri']
MODEL = MODEL_list[model]

## hyperparameters of all models
## we simply copy all the hyper-parameters for MBA same as the Amazon dataset
# learning rate, no big change 
LR_list = [[0.05, 0.0002,  0.001,  0.0001,  0.0001, 0.0001,  0.005,  0.0005, 0.0005, 0.0005, 0.0005, 0.005,],
           [0.02, 0.00001, 0.0002, 0.00005, 0.0001, 0.00002, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,],
           [0.05, 0.0002,  0.001,  0.0001,  0.0001, 0.0001,  0.005,  0.0005, 0.0005, 0.0005, 0.0005, 0.005,],
           [0.05, 0.0002,  0.001,  0.0001,  0.0001, 0.0001,  0.005,  0.0005, 0.0005, 0.0005, 0.0005, 0.005,], ]

# regularization factor, this is quite a tunable hyperparameter
LAMDA_list = [[0.02, 0, 0.05, 0.001, 0.02, 0.0002, 0.02, 0.005, 0.02, 0.02, 0.02, 0.02,],
              [0.01, 0, 0.02, 0.02,  0.01, 0.05,   0.02, 0.01,  0.1,  0.05, 0.1, 0.02,],
              [0.02, 0, 0.05, 0.001, 0.02, 0.0002, 0.02, 0.005, 0.02, 0.02, 0.02, 0.02,],
              [0.02, 0, 0.05, 0.001, 0.02, 0.0002, 0.02, 0.005, 0.02, 0.02, 0.02, 0.02,],]

# layer number
LAYER_list = [[0, 4, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2],
              [0, 4, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2],
              [0, 4, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1], # change to 1 layer for lightgcn_tri
              [0, 4, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1]] # 4*12, this always no change
LR = LR_list[dataset][model]
LAMDA = LAMDA_list[dataset][model]
LAYER = LAYER_list[dataset][model]
# dimensionality of the embedding layer
EMB_list = [pred_dim, int(pred_dim/2), int(pred_dim/(LAYER+1)), int(pred_dim/(LAYER+1)), int(pred_dim/(LAYER+1)), int(pred_dim/(LAYER+1)), pred_dim, int(pred_dim/(LAYER+1)), pred_dim, pred_dim, pred_dim, pred_dim,]
EMB_DIM = EMB_list[model]
BATCH_SIZE = 40000 # 10000
TEST_USER_BATCH_list = [4096, 1024, 512, 20620] # select all users for MBA: 4297; Instacart: 20620, originally 512 and 4096
TEST_USER_BATCH = TEST_USER_BATCH_list[dataset]
N_EPOCH = 300 # 200
IF_PRETRAIN = [False, True][0]
TEST_VALIDATION = 'Validation'  # can be changed automatically
TOP_K = [2, 5, 10, 20, 50, 100]

## hyperparameters for LCFN and LGCN
FREQUENCY_USER_list = [100, 300, 100, 100]
FREQUENCY_ITEM_list = [50, 200, 50, 50]
FREQUENCY_USER = FREQUENCY_USER_list[dataset] # I think both these no use for 1-d LGCN
FREQUENCY_ITEM = FREQUENCY_ITEM_list[dataset]

## hyperparameters for LGCN
FREQUENCY = 128
KEEP_PORB = 0.9
SAMPLE_RATE = 1
GRAPH_CONV = ['1D', '2D_graph', '2D_hyper_graph'][0] # 2
PREDICTION = ['InnerProduct', 'MLP3'][0]
LOSS_FUNCTION = ['BPR', 'CrossEntropy', 'MSE'][0]
GENERALIZATION = ['Regularization', 'DropOut', 'Regularization+DropOut', 'L2Norm'][0]
OPTIMIZATION = ['SGD', 'Adagrad', 'RMSProp', 'Adam'][2] # light: 3
IF_TRASFORMATION = [False, True][0]                           # 0 for not having transformation matrix,1 for having
ACTIVATION = ['None', 'Tanh', 'Sigmoid', 'ReLU'][0]          # select the activation function
POOLING = ['Concat', 'Sum', 'Max', 'Product', 'MLP3'][1]    # select the pooling strategy, the layer of mlp is also changable
if POOLING == 'Concat': EMB_DIM = int(pred_dim/(LAYER+1))

## parameters about model setting (selective for model LGCN)
PROP_DIM = 128
PROP_EMB = ['RM', 'SF', 'PE'][1]
IF_NORM = [False, True][0]

all_para = [GPU_INDEX, DATASET, MODEL, LR, LAMDA, LAYER, EMB_DIM, BATCH_SIZE, TEST_USER_BATCH, N_EPOCH, IF_PRETRAIN,
            TEST_VALIDATION, TOP_K, FREQUENCY_USER, FREQUENCY_ITEM, FREQUENCY, KEEP_PORB, SAMPLE_RATE, GRAPH_CONV,
            PREDICTION, LOSS_FUNCTION, GENERALIZATION, OPTIMIZATION, IF_TRASFORMATION, ACTIVATION, POOLING, PROP_DIM,
            PROP_EMB, IF_NORM] # last 3 are ignored