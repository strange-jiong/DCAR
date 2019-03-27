import tensorflow as tf
import copy

class Model(object):
    def __init__(self, config, user_N, item_N, train_user_item_matrix, ae_beta, dims, topk):

        self.config = config
        self.topk = topk
        self.user_N = user_N
        self.item_N = item_N
        self.train_user_item_matrix = tf.constant(train_user_item_matrix)
        self.train_item_user_matrix = train_user_item_matrix.T
        self.label = tf.placeholder(tf.int32, shape=[None, 1])

        # self.ae_beta_user_item_matrix = copy.deepcopy(self.train_user_item_matrix)
        # self.ae_beta_user_item_matrix[self.ae_beta_user_item_matrix > 0] = 1
        # self.ae_beta_user_item_matrix = self.ae_beta_user_item_matrix*ae_beta
        # self.ae_beta_item_user_matrix = self.ae_beta_user_item_matrix.T

        self.ae_beta_user_item_matrix = self.train_user_item_matrix*ae_beta
        self.ae_beta_item_user_matrix = tf.transpose(self.ae_beta_user_item_matrix)

        self.user_batch_idx = tf.placeholder(tf.int32, shape=[None])
        self.item_batch_idx = tf.placeholder(tf.int32, shape=[None])
        self.user_X = tf.nn.embedding_lookup(
            self.train_user_item_matrix, self.user_batch_idx)
        self.user_new_X = tf.nn.embedding_lookup(
            self.train_user_item_matrix, self.user_batch_idx)
        self.user_ae_beta = tf.nn.embedding_lookup(
            self.ae_beta_user_item_matrix, self.user_batch_idx)

        self.item_X = tf.nn.embedding_lookup(
            self.train_item_user_matrix, self.item_batch_idx)
        self.item_new_X = tf.nn.embedding_lookup(
            self.train_item_user_matrix, self.item_batch_idx)
        self.item_ae_beta = tf.nn.embedding_lookup(
            self.ae_beta_item_user_matrix, self.item_batch_idx)

        ############# define autoencoders ##########
        self.user_layers = len(config.user_struct)
        self.user_struct = self.config.user_struct
        self.item_layers = len(config.item_struct)
        self.item_struct = self.config.item_struct
        self.W = {}
        self.b = {}

        self.mlp_W = {}
        self.mlp_b = {}
        self.mlp_layers = config.mlp_layers

        user_struct = self.user_struct
        # two autoencoder
        # user autoencoder
        # encode module
        for i in range(self.user_layers-1):
            name_W = "user_encoder_W_"+str(i)
            name_b = "user_encoder_b_"+str(i)
            self.W[name_W] = tf.get_variable(
                name_W, [user_struct[i], user_struct[i+1]], initializer=tf.contrib.layers.xavier_initializer())
            self.b[name_b] = tf.get_variable(
                name_b, [user_struct[i+1]], initializer=tf.zeros_initializer())

        # decode module
        user_struct.reverse()

        for i in range(self.user_layers-1):
            name_W = "user_decoder_W_"+str(i)
            name_b = "user_decoder_b_"+str(i)
            self.W[name_W] = tf.get_variable(
                name_W, [user_struct[i], user_struct[i+1]], initializer=tf.contrib.layers.xavier_initializer())
            self.b[name_b] = tf.get_variable(
                name_b, [user_struct[i+1]], initializer=tf.zeros_initializer())
        user_struct.reverse()

        # item autoencoder
        item_struct = self.item_struct
        for i in range(self.item_layers-1):
            name_W = "item_encoder_W_"+str(i)
            name_b = "item_encoder_b_"+str(i)
            self.W[name_W] = tf.get_variable(
                name_W, [item_struct[i], item_struct[i+1]], initializer=tf.contrib.layers.xavier_initializer())
            self.b[name_b] = tf.get_variable(
                name_b, [item_struct[i+1]], initializer=tf.zeros_initializer())

        item_struct.reverse()
        for i in range(self.item_layers-1):
            name_W = "item_decoder_W_"+str(i)
            name_b = "item_decoder_b_"+str(i)
            self.W[name_W] = tf.get_variable(
                name_W, [item_struct[i], item_struct[i+1]], initializer=tf.contrib.layers.xavier_initializer())
            self.b[name_b] = tf.get_variable(
                name_b, [item_struct[i+1]], initializer=tf.zeros_initializer())
        item_struct.reverse()

        ########## define input #############
        # self.user = tf.placeholder(tf.int32)
        # self.item = tf.placeholder(tf.int32)

        # self.user_X = tf.placeholder(tf.float32, shape=[None, config.struct[0]])
        # self.item_X = tf.placeholder(tf.float32,shape=[None,config.struct[0]])

        # self.user_embedding=tf.nn.embedding_lookup(self.)

        self.make_user_autoencoder_compute_graph()
        self.make_item_autoencoder_compute_graph()
        self.make_autoencoder_loss()

        # compute geadient for deep autoencoder

        self.train_opt_ae = tf.train.AdamOptimizer(
            config.ae_learning_rate).minimize(self.loss_ae)

        # MLP for predict rating
        self.MLP = {}

        for i in range(len(self.mlp_layers)-1):
            name_W = "mlp_W_"+str(i)
            name_b = "mlp_b_"+str(i)
            self.mlp_W[name_W] = tf.get_variable(
                name_W, [self.mlp_layers[i], self.mlp_layers[i+1]], initializer=tf.contrib.layers.xavier_initializer())
            self.mlp_b[name_b] = tf.get_variable(
                name_b, [self.mlp_layers[i+1]], initializer=tf.zeros_initializer())
        # self.MLP['W']=tf.get_variable()
        self.make_mlp_compute_graph()
        self.predict_all()

        self.mlp_loss = self.make_mlp_loss()
        self.train_opt_mlp = tf.train.AdamOptimizer(
            config.mlp_learning_rate).minimize(self.mlp_loss)

    def make_mlp_compute_graph(self):
        """
        construct mlp graph
        """

        self.user_item_interaction = tf.concat([tf.multiply(self.user_Y, self.item_Y),self.user_Y,self.item_Y],axis=1)

        def compress(X):

            for i in range(len(self.mlp_layers)-1):
                if i == len(self.mlp_layers)-2:
                    name_W = "mlp_W_" + str(i)
                    name_b = "mlp_b_" + str(i)
                    # X = tf.nn.sigmoid(
                    #     tf.matmul(X, self.mlp_W[name_W]))
                    X = tf.matmul(X, self.mlp_W[name_W])  #+self.mlp_b[name_b]
                    # X = tf.nn.relu(
                    # tf.matmul(X, self.mlp_W[name_W])+self.mlp_b[name_b])
                    return X

                name_W = "mlp_W_" + str(i)
                name_b = "mlp_b_" + str(i)
                X = tf.nn.relu(
                    tf.matmul(X, self.mlp_W[name_W])+self.mlp_b[name_b])

                # X = tf.nn.sigmoid(
                #     tf.matmul(X, self.mlp_W[name_W]))

                # X = tf.matmul(X, self.mlp_W[name_W])

            return X

        self.user_item_predict = compress(self.user_item_interaction)

    def make_mlp_loss(self):
        """
        cross entroy loss

        """
        self.mlp_loss_func = tf.nn.sigmoid_cross_entropy_with_logits
        loss = tf.reduce_sum(self.mlp_loss_func(
            labels=tf.cast(self.label, tf.float32), logits=self.user_item_predict, name='mlp_loss'))
        return loss

    def predict(self):
        with tf.name_scope("evaluation"):
            self.logit_dense = tf.reshape(self.user_item_predict, [-1])
            _, self.indice = tf.nn.top_k(
                tf.sigmoid(self.logit_dense), self.topk)

        return self.indice

    def predict_all(self):
        self.logit_dense = tf.reshape(self.user_item_predict, [-1])
        self.predict_value = tf.sigmoid(self.logit_dense)
        # self.predict_value=self.logit_dense
        return self.predict_value

    def make_user_autoencoder_compute_graph(self):
        with tf.name_scope("user_autoencoder"):
            def encoder(X):
                for i in range(self.user_layers-1):
                    name_W = "user_encoder_W_"+str(i)
                    name_b = "user_encoder_b_"+str(i)
                    X = tf.nn.tanh(tf.matmul(X, self.W[name_W])+self.b[name_b])
                return X

            def decoder(X):
                for i in range(self.user_layers-1):
                    name_W = "user_decoder_W_"+str(i)
                    name_b = "user_decoder_b_"+str(i)
                    X = tf.nn.tanh(tf.matmul(X, self.W[name_W])+self.b[name_b])
                return X

            self.user_Y = encoder(self.user_X)
            self.user_X_reconstruct = decoder(self.user_Y)

    def make_item_autoencoder_compute_graph(self):
        with tf.name_scope("item_autoencoder"):
            def encoder(X):
                for i in range(self.item_layers-1):
                    name_W = "item_encoder_W_"+str(i)
                    name_b = "item_encoder_b_"+str(i)
                    X = tf.nn.tanh(tf.matmul(X, self.W[name_W])+self.b[name_b])
                    # print(X.shape)
                return X

            def decoder(X):
                for i in range(self.item_layers-1):
                    name_W = "item_decoder_W_"+str(i)
                    name_b = "item_decoder_b_"+str(i)
                    X = tf.nn.tanh(tf.matmul(X, self.W[name_W])+self.b[name_b])
                return X

            self.item_Y = encoder(self.item_X)
            self.item_X_reconstruct = decoder(self.item_Y)

    def make_autoencoder_loss(self):
        def get_autoencoders_loss(X, newX, X_beta):
            return tf.reduce_sum(tf.pow(tf.multiply((newX-X), X_beta), 2))

        def get_reg_loss(weights, biases):
            reg = tf.add_n([tf.nn.l2_loss(w) for w in weights.values()])
            reg += tf.add_n([tf.nn.l2_loss(b) for b in biases.values()])
            return reg

        # loss_autoencoder = get_autoencoders_loss(self.X_nex, self.user_X_reconstruct)
        user_loss_autoencoder = get_autoencoders_loss(
            self.user_new_X, self.user_X_reconstruct, self.user_ae_beta)
        item_loss_autoencoder = get_autoencoders_loss(
            self.item_new_X, self.item_X_reconstruct, self.item_ae_beta)

        loss_reg = get_reg_loss(self.W, self.b)
        self.loss_ae = self.config.alpha * \
            (user_loss_autoencoder+item_loss_autoencoder)+self.config.reg*loss_reg
