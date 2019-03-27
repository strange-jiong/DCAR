import sys


class Config(object):
    def __init__(self, dataset):
        # 对不同数据集的配置
        if dataset == "ml-1m":

            self.user_struct = [None, 1000, 500, None]
            self.item_struct = [None, 1000, 500, None]
            self.reg = 10
            self.alpha = 1
            self.ae_beta = 20

            self.mlp_layers = [None,64, 1]

            # parameters for training
            self.batch_size = 512
            self.num_sampled = 10
            self.max_iters = 300000
            self.sg_learning_rate = 1e-4
            self.ae_learning_rate = 1e-4
            self.mlp_learning_rate = 1e-4
        elif dataset == "ml-100k":

            self.user_struct = [None, 1000, 500, None]
            self.item_struct = [None, 1000, 500, None]
            self.reg = 10
            self.alpha = 1
            self.ae_beta = 20

            self.mlp_layers = [None, 64, 1]

            # parameters for training
            self.batch_size = 512
            self.num_sampled = 10
            self.max_iters = 300000
            self.sg_learning_rate = 1e-4
            self.ae_learning_rate = 1e-4
            self.mlp_learning_rate = 1e-4

        elif dataset == "ml-1m-test":
            self.user_struct = [None, 1000, 500, None]
            self.item_struct = [None, 1000, 500, None]
            self.reg = 10
            self.alpha = 1

            self.mlp_layers = [None, 1]

            # parameters for training
            self.batch_size = 5
            self.num_sampled = 10
            self.max_iters = 2000
            self.sg_learning_rate = 1e-4
            self.ae_learning_rate = 1e-4
            self.mlp_learning_rate = 1e-4

        elif dataset == "AMusic":
            self.user_struct = [None, 1000, 500, None]
            self.item_struct = [None, 1000, 500, None]
            self.reg = 10
            self.alpha = 1
            self.ae_beta= 20

            self.mlp_layers = [None,64, 1]

            # parameters for training
            self.batch_size = 256
            self.num_sampled = 10
            self.max_iters = 200000
            self.sg_learning_rate = 1e-4
            self.ae_learning_rate = 1e-4
            self.mlp_learning_rate = 1e-4

        elif dataset == "ABeauty":
            self.user_struct = [None, 1000, 500, None]
            self.item_struct = [None, 1000, 500, None]
            self.reg = 10
            self.alpha = 1
            self.ae_beta=30

            self.mlp_layers = [None, 1]

            # parameters for training
            self.batch_size = 512
            self.num_sampled = 10
            self.max_iters = 20000
            self.sg_learning_rate = 1e-4
            self.ae_learning_rate = 1e-4
            self.mlp_learning_rate = 1e-4
        elif dataset == "AGrocery":
            self.user_struct = [None, 1000, 500, None]
            self.item_struct = [None, 1000, 500, None]
            self.reg = 10
            self.alpha = 1
            self.ae_beta=30

            self.mlp_layers = [None, 1]

            # parameters for training
            self.batch_size = 512
            self.num_sampled = 10
            self.max_iters = 20000
            self.sg_learning_rate = 1e-4
            self.ae_learning_rate = 1e-4
            self.mlp_learning_rate = 1e-4
        elif dataset == "AToys":
            self.user_struct = [None, 1000, 500, None]
            self.item_struct = [None, 1000, 500, None]
            self.reg = 10
            self.alpha = 1
            self.ae_beta= 1

            self.mlp_layers = [None, 1]

            # parameters for training
            self.batch_size = 512
            self.num_sampled = 10
            self.max_iters = 200000
            self.sg_learning_rate = 1e-4
            self.ae_learning_rate = 1e-4
            self.mlp_learning_rate = 1e-4
        elif dataset == "ABaby":
            self.user_struct = [None, 1000, 500, None]
            self.item_struct = [None, 1000, 500, None]
            self.reg = 10
            self.alpha = 1
            self.ae_beta = 50

            self.mlp_layers = [None, 1]

            # parameters for training
            self.batch_size = 512
            self.num_sampled = 10
            self.max_iters = 20000
            self.sg_learning_rate = 1e-4
            self.ae_learning_rate = 1e-4
            self.mlp_learning_rate = 1e-4
        else:
            print("config Current data set is not support!")
            sys.exit()
