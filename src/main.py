import argparse
import tensorflow as tf

from model import Model
from dataset import DataSet
from evaluation import evaluation
import heapq
import numpy as np
from config import *
import time

from collections import defaultdict

evaluation = evaluation()

# set parameters
tf.app.flags.DEFINE_string("dataset", "ml-1m-test", "dataset description")
tf.app.flags.DEFINE_integer("dimension", 128, "embedding dimensions")
# tf.app.flags.DEFINE_integer("batch_size",2, "size of mini-batch")
tf.app.flags.DEFINE_integer("epochs", 10, "number of epoch")

tf.app.flags.DEFINE_integer("window_size", 3, "size of window")
tf.app.flags.DEFINE_integer("walk_length", 80, "length of walk")
tf.app.flags.DEFINE_integer("negNum", 5, "number of negative examples")
tf.app.flags.DEFINE_integer("topK", 50, "number of hitRatio")

FLAGS = tf.app.flags.FLAGS


def read_graph():
    G = nx.read_edgelist(args.train_file, nodetype=int, data=(
        ('weight', float),), create_using=nx.DiGraph())
    G = G.to_undirected()
    return G


# def generate_batch_iter():


def evaluate(sess,model,testNeg,top_k_list=None):
    HR = []
    NDCG = []

    HR_dict = defaultdict(lambda: [])
    NDCG_dict = defaultdict(lambda: [])
    
    test_user = testNeg[0]
    test_item = testNeg[1]
    for i in range(len(test_user)):
        target_item = test_item[i][0]
        feed_dict = {model.user_batch_idx: test_user[i],
                        model.item_batch_idx: test_item[i]}
        # model.predict_all()
        predict = sess.run(model.predict_value, feed_dict=feed_dict)

        # 这里计算topk的时候，直接使用tf.nn.top_k 不用得到 predict value 再排序

        item_score_dict = {}
        for j in range(len(test_item[i])):
            item = test_item[i][j]
            item_score_dict[item] = predict[j]

        ranklist = heapq.nlargest(
            FLAGS.topK, item_score_dict, key=item_score_dict.get)
        if top_k_list:
            for i in top_k_list:                        
                tmp_hr = evaluation.hit(target_item, ranklist[:i])
                tmp_ndcg = evaluation.ndcg(target_item, ranklist[:i])
                HR_dict['HR'+str(i)].append(tmp_hr)
                NDCG_dict['NDCG'+str(i)].append(tmp_ndcg)
        else:
            tmp_hr = evaluation.hit(target_item, ranklist)
            tmp_ndcg = evaluation.ndcg(target_item, ranklist)
            HR.append(tmp_hr)
            NDCG.append(tmp_ndcg)
    if top_k_list:
        for i in top_k_list:
            print('topK', i, ' ', "HR:{}, NDCG:{} ".format(
                np.mean(HR_dict['HR'+str(i)]), np.mean(NDCG_dict['NDCG'+str(i)])))
    else:
        print("HR:{}, NDCG:{} ".format(np.mean(HR), np.mean(NDCG)))



    

def main():
    config = Config(FLAGS.dataset)
    dims = FLAGS.dimension

    # use original rating matrix
    data_rating = DataSet(FLAGS.dataset)
    train_user_item_matrix = data_rating.get_rateing_train_matrix()
    # ae_beta = train_user_item_matrix* config.ae_beta
    # train_item_user_matrix = train_user_item_matrix.transpose()

    train_u, train_i, train_r = data_rating.getInstances(FLAGS.negNum)
    train_len = len(train_u)
    shuffled_idx = np.random.permutation(np.arange(train_len))
    train_u = train_u[shuffled_idx]
    train_i = train_i[shuffled_idx]
    train_r = train_r[shuffled_idx]

    # generate PPMI matrix 暂时先放着，最后加上， 可以和 original rating matrix 形成对比

    # nx_G = read_graph()
    # G = randomWalks.Graph(nx_G)
    # G.preprocess_transition_probs()
    # walks = G.simulate_walks(args.num_walks, args.walk_length)
    # ppmi_matrix = SPPMI.calculatePMIfromWalks(
    #     walks, args.window_size, args.walk_length, args.user_number, args.item_number)

    user_N, item_N = train_user_item_matrix.shape

    config.user_struct[0] = item_N
    config.user_struct[-1] = dims

    config.item_struct[0] = user_N
    config.item_struct[-1] = dims
    config.mlp_layers[0] = dims*3
    testNeg = data_rating.getTestNeg(data_rating.test, 99)
    model = Model(config, user_N, item_N,
                  train_user_item_matrix, config.ae_beta, dims, FLAGS.topK)

    # train 还没有完成
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)
    batch_size = config.batch_size
    max_iters = config.max_iters
    embedding_result = None

    idx = 0

    print_every_k_iterations = 500

    start = time.time()

    total_loss = 0
    loss_autoencoder = 0
    loss_mlp = 0

    for iter_cnt in range(max_iters):
        idx += 1
        # train for autoencoder
        # if iter_cnt < 5000:
        user_start_idx = np.random.randint(0, user_N-batch_size)
        user_batch_idx = np.array(
            range(user_start_idx, user_start_idx+batch_size))
        user_batch_idx = np.random.permutation(user_batch_idx)

        item_start_idx = np.random.randint(0, item_N-batch_size)
        item_batch_idx = np.array(
            range(item_start_idx, item_start_idx+batch_size))
        item_batch_idx = np.random.permutation(item_batch_idx)

        # print("item_start_idx", item_batch_idx.shape)
        # batch_user = train_user_item_matrix[batch_idx]

        feed_dict = {model.user_batch_idx: user_batch_idx,
                     model.item_batch_idx: item_batch_idx}
        _, loss_autoencoder_value = sess.run(
            [model.train_opt_ae, model.loss_ae], feed_dict=feed_dict)
        loss_autoencoder += loss_autoencoder_value

        # train for mlp prediction

        interaction_start_idx = np.random.randint(0, train_len-batch_size)
        interaction_batch_idx = np.array(
            range(interaction_start_idx, interaction_start_idx+batch_size))
        interaction_user_batch_idx = train_u[interaction_batch_idx]
        interaction_item_batch_idx = train_i[interaction_batch_idx]
        interaction_rate_batch_idx = train_r[interaction_batch_idx].reshape(
            -1, 1)

        # print("interaction_user_batch_idx", interaction_user_batch_idx.shape)
        # print("interaction_item_batch_idx", interaction_item_batch_idx.shape)
        # print("interaction_rate_batch_idx", interaction_rate_batch_idx.shape)

        feed_dict = {model.user_batch_idx: interaction_user_batch_idx,
                     model.item_batch_idx: interaction_item_batch_idx,
                     model.label: interaction_rate_batch_idx}
        _, loss_mlp_value = sess.run(
            [model.train_opt_mlp, model.mlp_loss], feed_dict=feed_dict)
        loss_mlp += loss_mlp_value

        if idx % print_every_k_iterations == 0:
            end = time.time()
            print("iterations:%d" %
                  (idx)+", time elapsed: %.2f," % (end-start),)
            total_loss = loss_mlp/idx+loss_autoencoder/idx
            print("loss； %.2f," % (total_loss),)

            evaluate(sess,model,testNeg, top_k_list=[1,2,3,4,5,6,7,8,9,10])

            # evaluate
            # evaluation.ndcg()



if __name__ == "__main__":
    main()
