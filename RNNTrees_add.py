import random
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import tree_qa as tree
import json
import util
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class Config(object):
    embed_size = 300
    label_size = 2
    max_epochs = 20
    lr = 0.003
    grad_clip = 5
    l2 = 3e-7


class RecursiveNetStaticGraph():
    def __init__(self, config):
        self.config = config

        # Load train data and build vocabulary
        self.train_data = tree.loadTrees('trees/train_sen3.txt',data_type='train')
        self.train_preds = np.load('trees/train_y_pred2.npy')
        self.train_ground_true = np.load('trees/train_y_true2.npy')
        self.dev_data = tree.loadTrees('trees/dev_sen3.txt',data_type='dev')
        self.dev_preds = np.load('trees/dev_y_pred2.npy')
        self.dev_ground_true = np.load('trees/dev_y_true2.npy')

        print('Loading embedding...')
        with open('dataset/word_dictionary.json', "r") as fh:
            self.word2id = json.load(fh)
            self.word2id['-LRB-'] = self.word2id['(']
            self.word2id['-RRB-'] = self.word2id[')']
            self.word2id['\'\''] = self.word2id['"']
            self.word2id['``'] = self.word2id['"']
            self.word2id['-LSB-'] = self.word2id['[']
            self.word2id['-RSB-'] = self.word2id[']']
            self.word2id['-LCB-'] = self.word2id['{']
            self.word2id['-RCB-'] = self.word2id['}']
        with open('dataset/word_emb.json','r') as fh:
            self.embed_mat = np.array(json.load(fh))
            print(self.embed_mat.shape)
        with open('dataset/dev_eval.json', "r") as fh:
            self.eval_file = json.load(fh)
        self.val_qid = np.load('dataset/dev_id.npy').astype(np.int32)

        # add input placeholders
        self.is_leaf_placeholder = tf.placeholder(tf.bool, (None), name='is_leaf_placeholder')
        self.left_children_placeholder = tf.placeholder(tf.int32, (None), name='left_children_placeholder')
        self.right_children_placeholder = tf.placeholder(tf.int32, (None), name='right_children_placeholder')
        self.node_word_indices_placeholder = tf.placeholder(tf.int32, (None), name='node_word_indices_placeholder')
        self.labels_placeholder = tf.placeholder(tf.int32, (None), name='labels_placeholder')
        self.node_id = tf.placeholder(tf.int32, (None), name='node_id')
        self.probes = tf.placeholder(tf.float32, (400, 2), name='probes')

        # add model variables
        with tf.variable_scope('Embeddings'):
            embeddings = tf.get_variable("embeddings", initializer=tf.constant(self.embed_mat, dtype=tf.float32), trainable=False)
        with tf.variable_scope('Composition'):
            W_leaf = tf.get_variable('W_leaf', [self.config.embed_size+2 , self.config.embed_size+2])
            W_node = tf.get_variable('W_node', [self.config.embed_size+2 , self.config.embed_size+2])
        b1 = tf.get_variable('b1', [1, self.config.embed_size+2])
        with tf.variable_scope('Projection'):
            U = tf.get_variable('U', [self.config.embed_size+2, self.config.label_size])
        bs = tf.get_variable('bs', [1, self.config.label_size])

        # build recursive graph

        tensor_array = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False)

        def embed_word(word_index, i):
            probe_feat = tf.expand_dims(tf.gather(self.probes, tf.gather(self.node_id, i)), 0)
            embed_feat = tf.expand_dims(tf.gather(embeddings, word_index), 0)
            # return probe_feat
            return tf.concat((embed_feat,probe_feat),axis=1)

        def combine_children_add(left_tensor, left_index, right_tensor, right_index):
            left_is_leaf = tf.gather(self.is_leaf_placeholder, left_index)
            x_left = tf.cond(left_is_leaf, lambda :tf.matmul(left_tensor, W_leaf), lambda :tf.matmul(left_tensor, W_node))
            right_is_leaf = tf.gather(self.is_leaf_placeholder, right_index)
            x_right = tf.cond(right_is_leaf, lambda :tf.matmul(right_tensor, W_leaf), lambda :tf.matmul(right_tensor, W_node))
            return tf.nn.relu(x_left + x_right + b1)

        def loop_body(tensor_array, i):
            node_is_leaf = tf.gather(self.is_leaf_placeholder, i)
            node_word_index = tf.gather(self.node_word_indices_placeholder, i)
            left_child = tf.gather(self.left_children_placeholder, i)
            right_child = tf.gather(self.right_children_placeholder, i)
            node_tensor = tf.cond(
                node_is_leaf,
                lambda: embed_word(node_word_index, i),
                lambda: combine_children_add(tensor_array.read(left_child), left_child,
                                         tensor_array.read(right_child), right_child))
            tensor_array = tensor_array.write(i, node_tensor)
            i = tf.add(i, 1)
            return tensor_array, i

        loop_cond = lambda tensor_array, i: tf.less(i, tf.squeeze(tf.shape(self.is_leaf_placeholder)))
        self.tensor_array, _ = tf.while_loop(loop_cond, loop_body, [tensor_array, 0], parallel_iterations=1)

        # add projection layer
        self.logits = tf.matmul(self.tensor_array.concat(), U) + bs
        self.max_index = tf.argmax(self.logits)

        # add loss layer
        regularization_loss = self.config.l2 * (tf.nn.l2_loss(W_leaf) + tf.nn.l2_loss(W_node)  + tf.nn.l2_loss(U))
        self.full_loss =  regularization_loss + tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_placeholder))

        # add training op
        self.opt = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
        grads = self.opt.compute_gradients(self.full_loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, self.config.grad_clip)
        self.train_op = self.opt.apply_gradients(zip(capped_grads, variables))

    def build_feed_dict(self, node, probes):
        nodes_list = []
        tree.leftTraverse(node, lambda node, args: args.append(node), nodes_list)
        node_to_index = OrderedDict()
        for i in range(len(nodes_list)):
            node_to_index[nodes_list[i]] = i
        feed_dict = {
            self.is_leaf_placeholder: [node.isLeaf for node in nodes_list],
            self.left_children_placeholder: [node_to_index[node.left] if
                                             not node.isLeaf else -1
                                             for node in nodes_list],
            self.right_children_placeholder: [node_to_index[node.right] if
                                              not node.isLeaf else -1
                                              for node in nodes_list],
            self.node_word_indices_placeholder: [self.word2id[node.word] if
                                                 node.word else -1
                                                 for node in nodes_list],
            self.labels_placeholder: [node.label for node in nodes_list],
            self.node_id: [node.id for node in nodes_list],
            self.probes: probes
        }
        return feed_dict,nodes_list

    def find_sted(self, max_index_value, nodes_list):
        def find_leftest(node):
            if node.left is None:
                return node
            else:
                return find_leftest(node.left)

        def find_rightest(node):
            if node.right is None:
                return node
            else:
                return find_rightest(node.right)

        result_node = nodes_list[max_index_value]
        st_point = find_leftest(result_node).id
        ed_point = find_rightest(result_node).id
        return [st_point, ed_point]

    def f1_cal(self, y_pred, y_true):
        y_pred = np.arange(y_pred[0],y_pred[1]+1)
        y_true = np.arange(y_true[0],y_true[1]+1)
        tp = len(list(set(y_pred).intersection(set(y_true))))
        precision = tp / len(y_pred)
        recall = tp / len(y_true)
        if precision + recall == 0:
            return 0
        else:
            return 2 * (precision * recall) / (precision + recall)

    def em_cal(self,y_pred, y_true):
        if y_pred[0]==y_true[0] and y_pred[1]==y_true[1]:
            return 1
        else:
            return 0

    def train(self, verbose=True):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.config.max_epochs):
                all_loss = []
                all_f1 = []
                all_em = []
                random.shuffle(self.train_data)
                for step, tree in enumerate(self.train_data):
                    feed_dict,nodes_list = self.build_feed_dict(tree.root,self.train_preds[tree.index,::])
                    loss_value, max_index_value, _ = sess.run([self.full_loss, self.max_index, self.train_op],feed_dict=feed_dict)
                    all_loss.append(loss_value)

                    # train validate
                    max_index_value = max_index_value[1]
                    y_pred_temp = self.find_sted(max_index_value, nodes_list)
                    y_true_temp = self.train_ground_true[tree.index,::]
                    f1_value = self.f1_cal(y_pred_temp,y_true_temp)
                    em_value = self.em_cal(y_pred_temp,y_true_temp)
                    all_f1.append(f1_value)
                    all_em.append(em_value)

                    if verbose:
                        last_train_str = "\r[epoch:%d/%d, steps:%d/%d] -loss: %.2f - f1: %.4f -em: %.4f" % \
                                         (epoch + 1, self.config.max_epochs, step + 1, len(self.train_data), np.mean(all_loss),
                                          np.mean(all_f1), np.mean(all_em))
                        print(last_train_str, end='      ', flush=True)

                # dev validate
                all_loss = []
                y_pred=[]
                val_qid_real=[]
                for step, tree in enumerate(self.dev_data):
                    feed_dict, nodes_list = self.build_feed_dict(tree.root, self.dev_preds[tree.index, ::])
                    loss_value, max_index_value, _ = sess.run([self.full_loss, self.max_index, self.train_op],
                                                              feed_dict=feed_dict)
                    all_loss.append(loss_value)
                    max_index_value = max_index_value[1]
                    y_pred_temp = self.find_sted(max_index_value, nodes_list)
                    y_true_temp = self.dev_ground_true[tree.index, ::]
                    val_qid_real.append(self.val_qid[tree.index])
                    y_pred.append([y_pred_temp[0]+y_true_temp[-1],y_pred_temp[1]+y_true_temp[-1]])

                    if verbose:
                        last_val_str = " [steps:%d/%d] -loss: %.2f" % \
                                         (step + 1, len(self.dev_data),
                                          np.mean(all_loss))
                        print(last_train_str+last_val_str, end='      ', flush=True)

                y_pred=np.array(y_pred)
                answer_dict, remapped_dict = util.convert_tokens(self.eval_file, val_qid_real, y_pred[:,0].tolist(),y_pred[:,1].tolist())
                metrics = util.evaluate(self.eval_file, answer_dict)
                print(last_train_str+last_val_str, " -EM: %.2f%%, -F1: %.2f%%" % (metrics['exact_match'], metrics['f1']), end=' ', flush=True)
                print('\n')



if __name__ == '__main__':
    config = Config()
    tree_graph = RecursiveNetStaticGraph(config)
    tree_graph.train(verbose=True)