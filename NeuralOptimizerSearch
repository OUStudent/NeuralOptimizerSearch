import argparse
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, TerminateOnNaN
import matplotlib.pyplot as plt
import time
import pickle
import logging
import math
import copy

import subprocess
import sys
import tensorflow as tf

import keras.backend as K

import networkx as nx

def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold=0,
                           total_steps=0,
                           start_lr=0.0,
                           target_lr=1e-3):
    # Cosine decay
    learning_rate = 0.5 * target_lr * (
            1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)

    learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate



class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, total_steps=0, warmup_steps=0, start_lr=0.0, target_lr=1e-3, hold=0):
        super(WarmupCosineDecay, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.global_step = 0
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.lrs = []

    def call(self):
        lr = lr_warmup_cosine_decay(global_step=self.global_step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold)
        self.global_step = self.global_step + 1
        return lr

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = self.model.optimizer.lr.numpy()
        self.lrs.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = lr_warmup_cosine_decay(global_step=self.global_step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold)
        K.set_value(self.model.optimizer.lr, lr + 1e-7)




(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_ind = list(range(0, 45000))
val_ind = list(range(45000, 50000))

AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128  # 512
IMG_SHAPE = 32

threshold = 0.40


def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SHAPE, IMG_SHAPE))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.
    return image, label


trainloader = tf.data.Dataset.from_tensor_slices(
    (x_train[train_ind], tf.keras.utils.to_categorical(y_train[train_ind])))
trainloader = (
    trainloader
    .map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

testloader = tf.data.Dataset.from_tensor_slices((x_train[val_ind], tf.keras.utils.to_categorical(y_train[val_ind])))
testloader = (
    testloader
    .map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)


class FiveEpochCallback(tf.keras.callbacks.Callback):

    def __init__(self, threshold, epoch=6):
        super(FiveEpochCallback, self).__init__()
        self.threshold = threshold
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.epoch:
            if logs['accuracy'] < self.threshold:  # 30
                self.model.stop_training = True


def ConvNet(act, base=32):
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(base, (3, 3), padding='same')(inputs)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(act)(x)
    x = tf.keras.layers.Conv2D(2 * base, (3, 3), strides=(2, 2), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(act)(x)
    x = tf.keras.layers.Conv2D(4 * base, (3, 3), strides=(2, 2), padding='same')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(act)(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


class DecayNode:

    def __init__(self, un_bin_perc, ID):
        self.un_bin_perc = un_bin_perc
        self.op_un_bin = np.random.choice(range(0, 2), p=un_bin_perc)
        self.un = np.random.choice(range(0, 13))
        self.bin = np.random.choice(range(0, 7))
        self.id = ID

    def call(self, in1, in2, name1, name2):

        eps = tf.keras.backend.epsilon()
        if self.op_un_bin == 0:  # unary
            if self.un == 0:
                total = tf.math.divide(1, (1 + tf.math.exp(-in1)))
                msg = "1/(1+exp(-{})".format(name1)
            elif self.un == 1:
                total = tf.math.divide(1, (1 + tf.math.abs(in1)))
                msg = "deriv_soft({})".format(name1)
            elif self.un == 2:
                total = tf.math.erf(in1)
                msg = "erf({})".format(name1)
            elif self.un == 3:
                total = tf.math.erfc(in1)
                msg = "erfc({})".format(name1)
            elif self.un == 4:
                total = tf.math.tanh(in1)
                msg = "tanh({})".format(name1)
            elif self.un == 5:
                total = tf.math.atan(in1)
                msg = "arctan({})".format(name1)
            elif self.un == 6:
                total = tf.math.bessel_i1e(in1)
                msg = "bessel_i1e({})".format(name1)
            elif self.un == 7:
                total = tf.math.pow(in1, 2)
                msg = "({})^2".format(name1)
            elif self.un == 8:
                total = tf.math.sqrt(tf.math.abs(in1))
                msg = "sqrt({})".format(name1)
            elif self.un == 9:
                total = tf.math.divide(in1, (1 + tf.math.abs(in1)))
                msg = "{}/(1+|{}|)".format(name1, name1)
            elif self.un == 10:
                total = 1 - tf.math.tanh(in1) * tf.math.tanh(in1)
                msg = "deriv_tanh({})".format(name1)
            elif self.un == 11:
                total = tf.math.sigmoid(in1) * (1 - tf.math.sigmoid(in1))
                msg = "deriv_sig({})".format(name1)
            elif self.un == 12:
                total = in1
                msg = "id({})".format(name1)
            else:
                total = in1
                msg = "un_error"
                print("UNARY: ERROR")
        else:  # binary
            if self.bin == 0:
                total = in1 + in2
                msg = "({})+({})".format(name1, name2)
            elif self.bin == 1:
                total = in1 - in2
                msg = "({})-({})".format(name1, name2)
            elif self.bin == 2:
                total = in1 * in2
                msg = "({})*({})".format(name1, name2)
            elif self.bin == 3:
                total = tf.math.maximum(in1, in2)
                msg = "max({}, {})".format(name1, name2)
            elif self.bin == 4:
                total = tf.math.minimum(in1, in2)
                msg = "min({}, {})".format(name1, name2)
            elif self.bin == 5:
                total = tf.math.divide(in1, (in2 + eps))
                msg = "({}) / ({})".format(name1, name2)
            elif self.bin == 6:
                total = tf.math.divide(in1, tf.math.sqrt(1 + tf.math.pow(in2, 2)))
                msg = "({}) / (sqrt(1+({})^2))".format(name1, name2)
            else:
                print("BINARY: ERROR")
                total = in1
                msg = "bin_err"

        return total, msg


class DecaySchedule:

    def __init__(self):
        self.node = None
        self.root = None
        self.un_bin_percs = [0.80, 0.20]  # unary - binary
        self.root_un_bin_perc = [0.80, 0.20]
        self.msg = None
        self.adj = None
        self.active_node = False
        self.phenotype = None
        self.T = None
        self.setup()

    def setup(self):
        """


        Until a valid DecaySchedule has been randomly created
           - Create Adjacency matrix
           - Why is it 16 x 16 ? I copied code and was too lazy to make it smaller, but it still follows paper

        """
        while True:
            self.adj = np.zeros(shape=(16, 16))
            # create Hidden State Decay Node
            self.node = DecayNode(un_bin_perc=self.un_bin_percs, ID=0)

            # decide if the node is unary or binary and adjust adjacency matrix to contain the indices to what arguments
            if self.node.op_un_bin == 0:  # unary
                idx = np.random.choice(range(0, 14))
                self.adj[14, idx] = 1
            else:
                idx = np.random.choice(range(0, 14))
                self.adj[14, idx] = 1
                idx = np.random.choice(np.concatenate((np.arange(0, idx), np.arange(idx + 1, 14))))
                self.adj[14, idx] = 2

            # create Root decay node
            self.root = DecayNode(un_bin_perc=self.root_un_bin_perc, ID=1)
            if self.root.op_un_bin == 0:  # unary
                if np.random.uniform(0, 1) > 0.8:
                    self.adj[-1, 14] = 1  # must be hidden state node
                else:
                    idx = np.random.choice(range(0, 15))
                    self.adj[-1, idx] = 1
            else:  # binary
                if np.random.uniform(0, 1) > 0.8:  # guarantee to contain hidden state node
                    if np.random.uniform(0, 1) > 0.5:  # order
                        self.adj[-1, 14] = 1  # hidden state node
                        idx = np.random.choice(range(0, 14))
                        self.adj[-1, idx] = 2
                    else:
                        self.adj[-1, 14] = 2  # hidden state node
                        idx = np.random.choice(range(0, 14))
                        self.adj[-1, idx] = 1
                else:  # not guaranteed
                    idx = np.random.choice(range(0, 15))
                    self.adj[-1, idx] = 1
                    idx = np.random.choice(np.concatenate((np.arange(0, idx), np.arange(idx + 1, 14))))
                    self.adj[-1, idx] = 2

            self.set_active()

            if not self.check_integrity():
                continue
            return

    def set_active(self):
        if self.adj[-1, 14] != 0:
            self.active_node = True

    def mutate(self, msgs, phenotypes):
        r = np.random.uniform(0, 1)
        if self.active_node:
            if np.random.uniform(0, 1) > 0.5:
                idx = 15
                node = self.root
            else:
                idx = 14
                node = self.node
        else:
            idx = 15
            node = self.root

        if idx == 15:
            c = np.random.choice(np.where(self.adj[idx] == 0)[0][0:-1])
        else:
            c = np.random.choice(np.where(self.adj[idx] == 0)[0][0:-2])
        if r >= 0.40:  # change operation
            if node.op_un_bin == 0:  # unary
                node.un = np.random.choice(np.concatenate((np.arange(0, node.un), np.arange(node.un + 1, 13))))
            else:
                node.bin = np.random.choice(np.concatenate((np.arange(0, node.bin), np.arange(node.bin + 1, 7))))
        elif r >= 0.15:  # change conn
            if node.op_un_bin == 0:  # unary
                self.adj[idx][self.adj[idx] == 1] = 0
                self.adj[idx][c] = 1
            else:
                if np.random.uniform(0, 1) <= 0.20:  # swap conn
                    idx1 = np.where(self.adj[idx] == 1)[0][0]
                    idx2 = np.where(self.adj[idx] == 2)[0][0]
                    self.adj[idx][idx1] = 2
                    self.adj[idx][idx2] = 1
                else:
                    if np.random.uniform(0, 1) < 0.5:  # change 1st conn
                        self.adj[idx][self.adj[idx] == 1] = 0
                        self.adj[idx][c] = 1
                    else:  # change 2nd conn
                        self.adj[idx][self.adj[idx] == 2] = 0
                        self.adj[idx][c] = 2
        else:  # un->bin , bin->un
            if node.op_un_bin == 0:  # unary -> binary
                node.op_un_bin = 1
                if np.random.uniform(0, 1) <= 0.5:  # add right conn
                    self.adj[idx][c] = 2
                else:
                    self.adj[idx][self.adj[idx] == 1] = 2
                    self.adj[idx][c] = 1
            else:  # binary -> unary
                node.op_un_bin = 0
                if np.random.uniform(0, 1) <= 0.5:  # delete right conn
                    self.adj[idx][self.adj[idx] == 2] = 0
                else:
                    self.adj[idx][self.adj[idx] == 1] = 0
                    self.adj[idx][self.adj[idx] == 2] = 1

        self.set_active()

        if not self.check_integrity():
            return False

        if self.msg in msgs:
            return False

        for pheno in phenotypes:
            if tf.norm(pheno - self.phenotype) <= 0.0001:
                return False
        return True

    def check_integrity(self):
        t = tf.convert_to_tensor(np.arange(0, 100), dtype=tf.float32)
        temp = self.T
        self.T = 100
        p = self.call(t).numpy()
        self.T = temp
        if np.any(p > 1):
            return False
        if np.any(p < 0):
            return False
        self.phenotype = p
        return True

    def call(self, t):
        a1 = 1.0 - tf.math.divide(t, self.T)
        a2 = tf.math.divide(t, self.T)
        a3 = 0.5 * (1 + tf.math.cos(tf.math.divide(math.pi * t, self.T)))
        a4 = 0.5 * (1 - tf.math.cos(tf.math.divide(math.pi * t, self.T)))
        a5 = 0.5 * (1 + tf.math.cos(tf.math.divide(2 * math.pi * t, self.T)))
        a6 = 0.5 * (1 - tf.math.cos(tf.math.divide(2 * math.pi * t, self.T)))
        a7 = 0.5 * (1 + tf.math.cos(math.pi * tf.math.divide(np.mod(2 * t, self.T), self.T)))
        a8 = 0.5 * (1 - tf.math.cos(math.pi * tf.math.divide(np.mod(2 * t, self.T), self.T)))
        a9 = 1 - tf.math.divide(np.mod(2 * t, self.T), self.T)
        a10 = tf.math.divide(np.mod(2 * t, self.T), self.T)
        a11 = tf.math.pow(0.01, tf.math.divide(t, self.T))
        a12 = 1 - tf.math.pow(0.01, tf.math.divide(t, self.T))
        beta_init = 0.95
        decay_rate = 1 - tf.math.divide((t + 1), self.T)
        beta_decay = beta_init * decay_rate
        a13 = tf.math.divide(beta_decay, ((1.0 - beta_init) + beta_decay))
        a14 = beta_init - tf.math.divide(beta_decay , ((1.0 - beta_init) + beta_decay))
        res = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, None]
        msgs = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12", "a13", "a14", None]

        if self.active_node:
            inds = np.where(self.adj[14] >= 1)[0]
            if len(inds) == 2:
                idx1 = np.where(self.adj[14] == 1)[0][0]
                idx2 = np.where(self.adj[14] == 2)[0][0]
                t, msg = self.node.call(res[idx1], res[idx2], msgs[idx1], msgs[idx2])

            else:
                t, msg = self.node.call(res[inds[0]], None, msgs[inds[0]], None)
            res[-1] = t
            msgs[-1] = msg

        inds = np.where(self.adj[-1] >= 1)[0]
        if len(inds) == 2:
            idx1 = np.where(self.adj[-1] == 1)[0][0]
            idx2 = np.where(self.adj[-1] == 2)[0][0]
            t, msg = self.root.call(res[idx1], res[idx2], msgs[idx1], msgs[idx2])
        else:
            t, msg = self.root.call(res[inds[0]], None, msgs[inds[0]], None)

        self.msg = msg
        return t




class Node:

    def __init__(self, un_bin_perc, ID):
        self.un_bin_perc = un_bin_perc
        self.op_un_bin = np.random.choice(range(0, 2), p=un_bin_perc)
        if np.random.uniform(0, 1) <= 0.20:  # identity
            self.un = -1
        else:
            self.un = np.random.choice(range(0, 25))
        self.bin = np.random.choice(range(0, 10))
        self.id = ID


# g, g^2, g^3, v hat, s hat, lamda hat, 1, 2, 10-6, 10-5w, 10-4w, 10-3w, sign(g), sign(v hat) -> 14
# moving avg v hat, moving avg s hat, movging avg lambda hat (QHM) -> 3
# mult beta v, mult beta s, mult beta lambda (AggMo) -> 3

from tensorflow.python.ops import control_flow_ops
# from tensorflow.python.ops import tf.math
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer


class Optimizer(optimizer.Optimizer):
    def __init__(self, nodes, learning_rate=0.001, use_locking=False, name="CustOpt", flip=False,
                 total_steps=0, warmup_steps=0, start_lr=0.0, hold=0, static=True, use_decay=False):
        super(Optimizer, self).__init__(use_locking, name)
        self._operands = nodes.node_priority
        self._nodes = nodes
        self._flip = flip
        self._iterations = 0
        self._msg = None
        self.type = nodes.type
        self.use_decay = use_decay
        self.decay = copy.deepcopy(nodes.decay)

        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.target_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.static = static
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        if self.static:
            self.lr = tf.constant(learning_rate, dtype=tf.float32)
            self.m = tf.constant(0.95, dtype=tf.float32)
        else:
            self.lr = tf.convert_to_tensor(
                tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32))
            t = np.concatenate((np.linspace(0.95, 0.85, self.warmup_steps),
                                np.linspace(0.85, 0.85, self.hold),
                                np.linspace(0.85, 0.95, self.total_steps - self.hold - self.warmup_steps)))
            self.m = tf.convert_to_tensor(tf.cast(t, dtype=tf.float32))
            # print(self.lr.dtype)

        self.decay_v = np.zeros(shape=[self.total_steps] + list(self.decay.shape))
        self.decay_msg = np.empty(shape=self.decay.shape, dtype=object)
        for i in range(0, len(self.decay)):
            for j in range(0, len(self.decay)):
                if self.decay[i][j] != None:
                    self.decay_v[:, i, j] = tf.cast(self.decay_call(np.arange(0, self.total_steps), self.total_steps,
                                                                    self.decay[i][j].adj,
                                                                    self.decay[i][j].node,
                                                                    self.decay[i][j].root,
                                                                    self.decay[i][j].active_node)[0],
                                                    dtype=tf.float32)

                    self.decay_msg[i][j] = self.decay[i][j].msg
        self.decay_v = tf.convert_to_tensor(
            self.decay_v)  # tf.Variable(self.decay_v, dtype=tf.float32, trainable=False)

    def _create_slots(self, var_list):
        hidden_state_nodes = self._nodes.node_priority[self._nodes.node_priority >= 20]
        for var in var_list:
            self._zeros_slot(var, "v", self._name)
            self._zeros_slot(var, "s", self._name)
            self._zeros_slot(var, "lam", self._name)
            if self.type == 1 or self.type == 2:
                self._zeros_slot(var, "mom", self._name)
            for node in hidden_state_nodes:
                if node == 25:
                    chosen_node = self._nodes.root
                else:
                    chosen_node = self._nodes.nodes[node - self._nodes.offset]
                if chosen_node.op_un_bin == 0:  # unary
                    if chosen_node.un == 22:
                        self._zeros_slot(var, "node{}_expAvg".format(node), self._name)
                    elif chosen_node.un == 23:
                        self._zeros_slot(var, "node{}_deltaChange".format(node), self._name)
                    elif chosen_node.un == 24:
                        self._zeros_slot(var, "node{}_MaxValue".format(node), self._name)
                else:
                    pass

    @staticmethod
    def decay_op(in1, in2, name1, name2, op_un_bin, un, bin):

        eps = tf.keras.backend.epsilon()
        if op_un_bin == 0:  # unary
            if un == 0:
                total = tf.math.divide(1, (1 + tf.math.exp(-in1)))
                msg = "1/(1+exp(-{})".format(name1)
            elif un == 1:
                total = tf.math.divide(1, (1 + tf.math.abs(in1)))
                msg = "deriv_soft({})".format(name1)
            elif un == 2:
                total = tf.math.erf(in1)
                msg = "erf({})".format(name1)
            elif un == 3:
                total = tf.math.erfc(in1)
                msg = "erfc({})".format(name1)
            elif un == 4:
                total = tf.math.tanh(in1)
                msg = "tanh({})".format(name1)
            elif un == 5:
                total = tf.math.atan(in1)
                msg = "arctan({})".format(name1)
            elif un == 6:
                total = tf.math.bessel_i1e(in1)
                msg = "bessel_i1e({})".format(name1)
            elif un == 7:
                total = tf.math.pow(in1, 2)
                msg = "({})^2".format(name1)
            elif un == 8:
                total = tf.math.sqrt(tf.math.abs(in1))
                msg = "sqrt({})".format(name1)
            elif un == 9:
                total = tf.math.divide(in1, (1 + tf.math.abs(in1)))
                msg = "{}/(1+|{}|)".format(name1, name1)
            elif un == 10:
                total = 1 - tf.math.tanh(in1) * tf.math.tanh(in1)
                msg = "deriv_tanh({})".format(name1)
            elif un == 11:
                total = tf.math.sigmoid(in1) * (1 - tf.math.sigmoid(in1))
                msg = "deriv_sig({})".format(name1)
            elif un == 12:
                total = in1
                msg = "id({})".format(name1)
            else:
                total = in1
                msg = "un_error"
                print("UNARY: ERROR")
        else:  # binary
            if bin == 0:
                total = in1 + in2
                msg = "({})+({})".format(name1, name2)
            elif bin == 1:
                total = in1 - in2
                msg = "({})-({})".format(name1, name2)
            elif bin == 2:
                total = in1 * in2
                msg = "({})*({})".format(name1, name2)
            elif bin == 3:
                total = tf.math.maximum(in1, in2)
                msg = "max({}, {})".format(name1, name2)
            elif bin == 4:
                total = tf.math.minimum(in1, in2)
                msg = "min({}, {})".format(name1, name2)
            elif bin == 5:
                total = tf.math.divide(in1, (in2 + eps))
                msg = "({}) / ({})".format(name1, name2)
            elif bin == 6:
                total = tf.math.divide(in1, tf.math.sqrt(1 + tf.math.pow(in2, 2)))
                msg = "({}) / (sqrt(1+({})^2))".format(name1, name2)
            else:
                print("BINARY: ERROR")
                total = in1
                msg = "bin_err"

        return total, msg

    def decay_call(self, t, T, adj, node, root, active_node):
        a1 = tf.cast(1.0 - tf.math.divide(t, T), dtype=tf.float32)
        a2 = tf.cast(tf.math.divide(t, T), dtype=tf.float32)
        a3 = tf.cast(0.5 * (1 + tf.math.cos(tf.math.divide(math.pi * t, T))), dtype=tf.float32)
        a4 = tf.cast(0.5 * (1 - tf.math.cos(tf.math.divide(math.pi * t, T))), dtype=tf.float32)
        a5 = tf.cast(0.5 * (1 + tf.math.cos(tf.math.divide(2 * math.pi * t, T))), dtype=tf.float32)
        a6 = tf.cast(0.5 * (1 - tf.math.cos(tf.math.divide(2 * math.pi * t, T))), dtype=tf.float32)
        a7 = tf.cast(0.5 * (1 + tf.math.cos(math.pi * tf.math.divide(np.mod(2 * t, T), T))), dtype=tf.float32)
        a8 = tf.cast(0.5 * (1 - tf.math.cos(math.pi * tf.math.divide(np.mod(2 * t, T), T))), dtype=tf.float32)
        a9 = tf.cast(1 - tf.math.divide(np.mod(2 * t, T), T), dtype=tf.float32)
        a10 = tf.cast(tf.math.divide(np.mod(2 * t, T), T), dtype=tf.float32)
        a11 = tf.cast(tf.math.pow(0.01, tf.math.divide(t, T)), dtype=tf.float32)
        a12 = tf.cast(1 - tf.math.pow(0.01, tf.math.divide(t, T)), dtype=tf.float32)
        beta_init = 0.95
        decay_rate = tf.cast(1 - tf.math.divide((t + 1), T), dtype=tf.float32)
        beta_decay = tf.cast(beta_init * decay_rate, dtype=tf.float32)
        a13 = tf.cast(tf.math.divide(beta_decay, ((1.0 - beta_init) + beta_decay)), dtype=tf.float32)
        a14 = beta_init -tf.cast(tf.math.divide( beta_decay, ((1.0 - beta_init) + beta_decay)), dtype=tf.float32)
        res = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, None]
        msgs = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12", "a13", "a14", None]

        if active_node:
            inds = np.where(adj[14] >= 1)[0]
            if inds.shape[0] == 2:
                idx1 = np.where(adj[14] == 1)[0][0]
                idx2 = np.where(adj[14] == 2)[0][0]
                t, msg = self.decay_op(res[idx1], res[idx2], msgs[idx1], msgs[idx2], node.op_un_bin, node.un, node.bin)

            else:
                t, msg = self.decay_op(res[inds[0]], None, msgs[inds[0]], None, node.op_un_bin, node.un, node.bin)
            res[-1] = t
            msgs[-1] = msg

        inds = np.where(adj[-1] >= 1)[0]
        if inds.shape[0] == 2:
            idx1 = np.where(adj[-1] == 1)[0][0]
            idx2 = np.where(adj[-1] == 2)[0][0]
            t, msg = self.decay_op(res[idx1], res[idx2], msgs[idx1], msgs[idx2], root.op_un_bin, root.un, root.bin)
        else:
            t, msg = self.decay_op(res[inds[0]], None, msgs[inds[0]], None, root.op_un_bin, root.un, root.bin)

        return t, msg

    @staticmethod
    def _binary(in1, in2, name1, name2, bin):

        eps = tf.keras.backend.epsilon()
        if bin == 0:
            total = in1 + in2
            msg = "({})+({})".format(name1, name2)
        elif bin == 1:
            total = in1 - in2
            msg = "({})-({})".format(name1, name2)
        elif bin == 2:
            total = in1 * in2
            msg = "({})*({})".format(name1, name2)
        elif bin == 3:
            total = tf.math.maximum(in1, in2)
            msg = "max({}, {})".format(name1, name2)
        elif bin == 4:
            total = tf.math.minimum(in1, in2)
            msg = "min({}, {})".format(name1, name2)
        elif bin == 5:
            total = tf.math.divide(in1, (in2 + eps))
            msg = "({}) / ({})".format(name1, name2)
        elif bin == 6:
            total = tf.math.divide(in1, tf.math.sqrt(1 + tf.math.pow(in2, 2)))
            msg = "({}) / (sqrt(1+({})^2))".format(name1, name2)
        elif bin == 7:
            total = tf.clip_by_value(in1, clip_value_min=-tf.math.abs(in2), clip_value_max=tf.math.abs(in2))
            msg = "clip({}, {})".format(name1, name2)
        elif bin == 8:
            total = 0.95 * in1 + (1 - 0.95) * in2
            msg = "(0.95*({})+0.05*({}))".format(name1, name2)
        elif bin == 9:
            total = tf.math.pow(tf.math.abs(in1), in2)
            msg = "(|{}|)^({})".format(name1, name2)
        else:
            print("BINARY: ERROR")
            total = in1
            msg = "bin_err"
        return total, msg

    @staticmethod
    def _unary(in1, name1, un):
        eps = tf.keras.backend.epsilon()
        if un == -1:
            total = in1
            msg = "id({})".format(name1)
        elif un == 0:
            total = - in1
            msg = "-({})".format(name1)
        elif un == 1:
            total = tf.math.log(tf.math.abs(in1) + eps)
            msg = "ln(|{}|)".format(name1)
        elif un == 2:
            total = tf.math.sqrt(tf.math.abs(in1))
            msg = "sqrt(|{}|)".format(name1)
        elif un == 3:
            total = tf.math.exp(in1)
            msg = "exp({})".format(name1)
        elif un == 4:
            total = tf.math.abs(in1)
            msg = "|{}|".format(name1)
        elif un == 5:
            total = tf.math.divide(1, (1 + tf.math.exp(-in1)))
            msg = "sig({})".format(name1)
        elif un == 6:
            total = tf.math.divide(1, (1 + tf.math.abs(in1)))
            msg = "1/(1+|{}|)".format(name1)
        elif un == 7:
            total = tf.math.log(tf.math.abs(1 + tf.math.exp(in1)) + eps)
            msg = "ln(|1+exp({})|)".format(name1)
        elif un == 8:
            total = tf.math.erf(in1)
            msg = "erf({})".format(name1)
        elif un == 9:
            total = tf.math.erfc(in1)
            msg = "erfc({})".format(name1)
        elif un == 10:
            total = tf.math.asinh(in1)
            msg = "arcsinh({})".format(name1)
        elif un == 11:
            total = tf.math.tanh(in1)
            msg = "tanh({})".format(name1)
        elif un == 12:
            total = tf.math.atan(in1)
            msg = "arctan({})".format(name1)
        elif un == 13:
            total = tf.math.divide(1, (in1 + eps))
            msg = "1/({})".format(name1)
        elif un == 14:
            total = tf.math.bessel_i1e(in1)
            msg = "bessel_i1e({})".format(name1)
        elif un == 15:
            total = tf.math.maximum(in1, 0)
            msg = "max({}, 0)".format(name1)
        elif un == 16:
            total = tf.math.minimum(in1, 0)
            msg = "min({}, 0)".format(name1)
        elif un == 17:
            total = tf.nn.dropout(in1, rate=0.5)
            msg = "drop({}, 0.5)".format(name1)
        elif un == 18:
            total = tf.nn.dropout(in1, rate=0.3)
            msg = "drop({}, 0.3)".format(name1)
        elif un == 19:
            total = tf.nn.dropout(in1, rate=0.1)
            msg = "drop({}, 0.1)".format(name1)
        elif un == 20:
            total = tf.clip_by_norm(in1, 1)
            msg = "norm({})".format(name1)
        elif un == 21:
            total = tf.math.divide(in1, (1 + tf.math.abs(in1)))
            msg = "{}/(1+|{}|)".format(name1, name1)
        else:
            total = in1
            msg = "un_error"
            print("UNARY: ERROR")
        return total, msg

    def _finish(self, update_ops, name_scope):
        t = self.t.assign_add(1.0)
        if self.static:
            return control_flow_ops.group(*update_ops + [t], name=name_scope)
        # lr = self.lr.assign(self.lr_warmup_cosine_decay())
        # vs = np.zeros(shape=self.decay.shape)
        """for i in range(0, len(self.decay)):
            for j in range(0, len(self.decay)):
                if self.decay[i][j] is not None:
                    vs[i][j] = self.decay_call(self.t, self.total_steps,
                                   self.decay[i][j].adj, self.decay[i][j].node, self.decay[i][j].root,
                                  self.decay[i][j].active_node)[0]

        v = self.decay_v.assign(vs)"""
        # self.decay_v[i][j] = tf.cast(self.decay_call(0, self.total_steps,
        #               self.decay[i][j].adj, self.decay[i][j].node, self.decay[i][j].root,
        #               self.decay[i][j].active_node)[0], dtype=tf.float32)
        # v = self.decay_v[i][j].assign(tf.cast(self.decay_call(0, self.total_steps,
        #               self.decay[i][j].adj, self.decay[i][j].node, self.decay[i][j].root,
        #               self.decay[i][j].active_node)[0], dtype=tf.float32))
        # vs.append(v)
        # tf.print(t, lr, d)
        return control_flow_ops.group(*update_ops + [t], name=name_scope)

    def lr_warmup_cosine_decay(self, t):
        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (
                1 + tf.math.cos(
            math.pi * (t - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        warmup_lr = self.target_lr * (t / self.warmup_steps)

        if self.hold > 0:
            learning_rate = tf.where(t > self.warmup_steps + self.hold,
                                     learning_rate, self.start_lr)

        learning_rate = tf.where(t < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self.lr, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]  # self._lr_t
        m_t = tf.cast(self.m, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]
        d = tf.cast(self.decay_v, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]
        b7 = tf.cast(0.70, var.dtype.base_dtype)
        b9 = tf.cast(0.9, var.dtype.base_dtype)
        b95 = tf.cast(0.95, var.dtype.base_dtype)
        b99 = tf.cast(0.99, var.dtype.base_dtype)
        b999 = tf.cast(0.999, var.dtype.base_dtype)
        b9999 = tf.cast(0.9999, var.dtype.base_dtype)
        one = tf.cast(1, var.dtype.base_dtype)
        two = tf.cast(2, var.dtype.base_dtype)
        three = tf.cast(3, var.dtype.base_dtype)

        node_res = {}
        msgs_res = {}
        for i in range(0, self._nodes.num_nodes + self._nodes.offset):
            node_res[i] = None
            msgs_res[i] = None
        # g, g^2, g^3, v hat, s hat, lamda hat, 1, 2, 10-6, 10-5w, 10-4w, 10-3w, sign(g), sign(v hat) -> 14
        # moving avg v hat, moving avg s hat, movging avg lambda hat (QHM) -> 3
        # mult beta v, mult beta s, mult beta lambda (AggMo) -> 3
        node_priority = self._nodes.node_priority
        sorted_terminal_nodes = np.sort(node_priority[node_priority < 20])
        hidden_state_nodes = node_priority[node_priority >= 20]
        node_res[0] = grad
        node_res[1] = tf.math.pow(grad, 2)
        node_res[2] = tf.math.pow(grad, 3)
        v = self.get_slot(var, "v")
        v1 = v.assign(b9 * v + (one - b9) * grad)
        node_res[3] = tf.math.divide(v1, one - tf.math.pow(b9, self.t + 1))
        s = self.get_slot(var, "s")
        s1 = s.assign(b99 * s + (one - b99) * node_res[1])
        node_res[4] = tf.math.divide(s1, one - tf.math.pow(b99, self.t + 1))
        lam = self.get_slot(var, "lam")
        lam1 = lam.assign(b999 * lam + (one - b999) * node_res[2])
        node_res[5] = tf.math.divide(lam1, one - tf.math.pow(b999, self.t + 1))

        msgs_res[0] = "g"
        msgs_res[1] = "g^2"
        msgs_res[2] = "g^3"
        msgs_res[3] = "v hat"
        msgs_res[4] = "s hat"
        msgs_res[5] = "lam hat"

        for node in sorted_terminal_nodes:
            if node_res[node] is None:
                if node == 6:
                    node_res[node] = tf.ones_like(var)
                    msgs_res[node] = "1"
                elif node == 7:
                    node_res[node] = 2 * tf.ones_like(var)
                    msgs_res[node] = "2"
                elif node == 8:
                    node_res[node] = 1e-6 * var
                    msgs_res[node] = "10-6w"
                elif node == 9:
                    node_res[node] = 1e-5 * var
                    msgs_res[node] = "10-5w"
                elif node == 10:
                    node_res[node] = 1e-4 * var
                    msgs_res[node] = "10-4w"
                elif node == 11:
                    node_res[node] = 1e-3 * var
                    msgs_res[node] = "10-3w"
                elif node == 12:
                    node_res[node] = tf.math.sign(node_res[0])
                    msgs_res[node] = "sign(g)"
                elif node == 13:
                    node_res[node] = tf.math.sign(node_res[3])
                    msgs_res[node] = "sign(v hat)"
                elif node == 14:
                    node_res[node] = (one - b7) * node_res[0] + b7 * node_res[3]
                    msgs_res[node] = "((1-0.7)*g+0.7*v hat)"
                elif node == 15:
                    node_res[node] = (one - b95) * node_res[1] + b95 * node_res[4]
                    msgs_res[node] = "((1-0.95)*g^2+0.95*s hat)"
                elif node == 16:
                    node_res[node] = (one - b99) * node_res[2] + b99 * node_res[5]
                    msgs_res[node] = "((1-0.99)*g^2+0.99*lam hat)"
                elif node == 17:
                    t0 = self.get_slot(var, "v")
                    t1 = - node_res[0]
                    t2 = b9 * t0 - node_res[0]
                    t3 = b999 * t0 - node_res[0]
                    node_res[node] = tf.math.divide(t1 + t2 + t3, three)
                    msgs_res[node] = "mean(beta*v-g)"
                elif node == 18:
                    t0 = self.get_slot(var, "s")
                    t1 = - node_res[1]
                    t2 = b99 * t0 - node_res[1]
                    t3 = b999 * t0 - node_res[1]
                    node_res[node] = tf.math.divide(t1 + t2 + t3, three)
                    msgs_res[node] = "mean(beta*s-g^2)"
                elif node == 19:
                    t0 = self.get_slot(var, "lam")
                    t1 = - node_res[2]
                    t2 = b999 * t0 - node_res[2]
                    t3 = b9999 * t0 - node_res[2]
                    node_res[node] = tf.math.divide(t1 + t2 + t3, three)
                    msgs_res[node] = "mean(beta*lam-g^3)"

        variables = [v, s, lam]
        for node in hidden_state_nodes:

            if node == 25:
                chosen_node = self._nodes.root
            else:
                chosen_node = self._nodes.nodes[node - self._nodes.offset]
            inds = np.where(self._nodes.adj[node] >= 1)[0]

            if len(inds) == 2:
                idx1 = np.where(self._nodes.adj[node] == 1)[0][0]
                idx2 = np.where(self._nodes.adj[node] == 2)[0][0]
                bin = chosen_node.bin
                r1 = node_res[idx1]
                r2 = node_res[idx2]
                # tf.print(node, idx1, idx2, r1, r2)
                m1, m2 = msgs_res[idx1], msgs_res[idx2]
                if self.use_decay:
                    if self.decay[node][idx1] != None:
                        """t, m = self.decay_call(self.t, self.total_steps, self.decay[node][idx1].adj,
                                        self.decay[node][idx1].node,
                                        self.decay[node][idx1].root,
                                        self.decay[node][idx1].active_node)"""
                        r1 = r1 * d[node][idx1]  # t
                        m = self.decay_msg[node][idx1]
                        m1 = "[{}][{}]".format(m, m1)
                    if self.decay[node][idx2] != None:
                        """t, m = self.decay_call(self.t, self.total_steps, self.decay[node][idx2].adj,
                                               self.decay[node][idx2].node,
                                               self.decay[node][idx2].root,
                                               self.decay[node][idx2].active_node)"""
                        r2 = r2 * d[node][idx2]  # t
                        m = self.decay_msg[node][idx2]
                        m2 = "[{}][{}]".format(m, m2)

                        # r2 *= self.decay[node][idx2].call(self.t)
                        # m2 = "[{}][{}]".format(self.decay[node][idx2].msg, m2)
                t, msg = self._binary(r1, r2, m1, m2, bin)
                node_res[node] = t
                msgs_res[node] = msg
            else:
                un = chosen_node.un
                r1 = node_res[inds[0]]
                # tf.print(node, inds[0], un, r1)
                if un == 22:
                    n = self.get_slot(var, "node{}_expAvg".format(node))
                    msg = "expMovAvg({})".format(msgs_res[inds[0]])
                    if self.use_decay:
                        if self.decay[node][inds[0]] != None:
                            """t, m = self.decay_call(self.t, self.total_steps, self.decay[node][inds[0]].adj,
                                                   self.decay[node][inds[0]].node,
                                                   self.decay[node][inds[0]].root,
                                                   self.decay[node][inds[0]].active_node)"""
                            r1 = r1 * d[node][inds[0]]  # t
                            m = self.decay_msg[node][inds[0]]
                            msg = "expMovAvg([{}][{}])".format(m, msgs_res[inds[0]])
                    n1 = n.assign(b95 * n + (one - b95) * r1)
                    variables.append(n)
                    t = tf.math.divide(n1, one - tf.math.pow(b95, self._iterations))
                elif un == 23:
                    msg = "deltaChange({})".format(msgs_res[inds[0]])
                    if self.use_decay:
                        if self.decay[node][inds[0]] != None:
                            """t, m = self.decay_call(self.t, self.total_steps, self.decay[node][inds[0]].adj,
                                                   self.decay[node][inds[0]].node,
                                                   self.decay[node][inds[0]].root,
                                                   self.decay[node][inds[0]].active_node)"""
                            r1 = r1 * d[node][inds[0]]  # t
                            m = self.decay_msg[node][inds[0]]
                            msg = "deltaChange([{}][{}])".format(m, msgs_res[inds[0]])
                    n = self.get_slot(var, "node{}_deltaChange".format(node))
                    t = n.assign(r1 - n)
                    variables.append(n)
                elif un == 24:
                    msg = "MaxValue({})".format(msgs_res[inds[0]])
                    if self.use_decay:
                        if self.decay[node][inds[0]] != None:
                            """t, m = self.decay_call(self.t, self.total_steps, self.decay[node][inds[0]].adj,
                                                   self.decay[node][inds[0]].node,
                                                   self.decay[node][inds[0]].root,
                                                   self.decay[node][inds[0]].active_node)"""
                            r1 = r1 * d[node][inds[0]]  # t
                            m = self.decay_msg[node][inds[0]]
                            msg = "MaxValue([{}][{}])".format(m, msgs_res[inds[0]])
                    n = self.get_slot(var, "node{}_MaxValue".format(node))
                    t = n.assign(tf.math.maximum(r1, n))
                    variables.append(n)
                else:
                    m = msgs_res[inds[0]]
                    if self.use_decay:
                        if self.decay[node][inds[0]] != None:
                            """t, md = self.decay_call(self.t, self.total_steps, self.decay[node][inds[0]].adj,
                                                   self.decay[node][inds[0]].node,
                                                   self.decay[node][inds[0]].root,
                                                   self.decay[node][inds[0]].active_node)"""
                            r1 = r1 * d[node][inds[0]]  # t
                            md = self.decay_msg[node][inds[0]]
                            m = "[{}][{}]".format(md, m)
                    t, msg = self._unary(r1, m, un)

                node_res[node] = t
                msgs_res[node] = msg

        update = node_res[node]
        msg = msgs_res[node]
        if self.type == 0:
            msg = "None ~ " + msg
            if self._flip:
                var_update = state_ops.assign_add(var, lr_t * update)
            else:
                var_update = state_ops.assign_add(var, - lr_t * update)
        elif self.type == 1:
            msg = "Mom ~ " + msg
            mom = self.get_slot(var, "mom")
            if self._flip:
                mom1 = mom.assign(m_t * mom + lr_t * update)
            else:
                mom1 = mom.assign(m_t * mom - lr_t * update)
            variables.append(mom)
            var_update = state_ops.assign_add(var, mom1)
        else:
            msg = "Nest ~ " + msg
            mom = self.get_slot(var, "mom")
            if self._flip:
                mom1 = mom.assign(m_t * mom + lr_t * update)
                var_update = state_ops.assign_add(var, mom1 + lr_t * update)
            else:
                mom1 = mom.assign(m_t * mom - lr_t * update)
                var_update = state_ops.assign_add(var, mom1 - lr_t * update)
            variables.append(mom)
        self.msg = msg

        final = [var_update] + variables
        return control_flow_ops.group(*final)



class OptimizerChrom:

    def __init__(self, label_smoothing=0.05):
        self.nodes = []
        self.root = None
        self.flip = False
        self.age = 0
        self.un_bin_percs = [  # un_bin_perc short for unary_binary_percents
            [0.75, 0.25],
            [0.75, 0.25],
            [0.75, 0.25],
            [0.75, 0.25],
            [0.75, 0.25],
        ]
        self.root_un_bin_perc = [0.25, 0.75]
        self.msg = None
        self.adj = None
        self.decay = None
        self.active_nodes = None
        self.node_priority = None
        self.label_smoothing = label_smoothing
        self.phenotype = None
        self.threshold = 0.10
        self.num_nodes = 5
        self.offset = 20
        self.flip = False
        self.type = None
        self.setup()

    def setup(self):
        self.offset = 20
        decay_prob = 0.15
        while True:
            self.nodes = []
            self.adj = np.zeros(
                shape=(self.offset + self.num_nodes + 1, self.offset + self.num_nodes + 1))  # 5 nodes, 20 operands
            self.decay = np.empty(shape=self.adj.shape, dtype=object)
            for i in range(0, self.num_nodes):
                self.nodes.append(Node(un_bin_perc=self.un_bin_percs[i], ID=i))
                if self.nodes[-1].op_un_bin == 0:  # unary
                    if np.random.uniform(0, 1) >= 0.20:  # operand
                        idx = np.random.choice(np.arange(0, self.offset))
                    else:  # hidden state node
                        idx = np.random.choice(
                            np.concatenate((np.arange(self.offset, self.offset + i), np.arange(self.offset + i + 1,
                                                                                               self.offset + self.num_nodes))))
                    self.adj[i + self.offset, idx] = 1
                    if np.random.uniform(0, 1) <= decay_prob:
                        self.decay[i + self.offset, idx] = DecaySchedule()
                else:  # binary

                    if np.random.uniform(0, 1) >= 0.20:  # operand
                        idx1 = np.random.choice(np.arange(0, self.offset))
                    else:  # hidden state node
                        idx1 = np.random.choice(np.concatenate((np.arange(self.offset, self.offset + i),
                                                                np.arange(self.offset + i + 1,
                                                                          self.offset + self.num_nodes))))
                    self.adj[i + self.offset, idx1] = 1
                    if np.random.uniform(0, 1) <= decay_prob:
                        self.decay[i + self.offset, idx1] = DecaySchedule()

                    if np.random.uniform(0, 1) >= 0.20:  # operand
                        if idx1 < self.offset:
                            idx2 = np.random.choice(
                                np.concatenate((np.arange(0, idx1), np.arange(idx1 + 1, self.offset))))
                        else:
                            idx2 = np.random.choice(np.arange(0, self.offset))
                    else:  # hidden state node
                        if idx1 < self.offset:  # first arg is operand
                            idx2 = np.random.choice(np.concatenate((np.arange(self.offset, self.offset + i),
                                                                    np.arange(self.offset + i + 1,
                                                                              self.offset + self.num_nodes))))
                        else:
                            idx2 = np.random.choice([ind for ind in np.arange(self.offset, self.offset + self.num_nodes)
                                                     if ind not in [i, idx1]])

                    self.adj[i + self.offset, idx2] = 2
                    if np.random.uniform(0, 1) <= decay_prob:
                        self.decay[i + self.offset, idx2] = DecaySchedule()

            self.root = Node(un_bin_perc=self.root_un_bin_perc, ID=self.num_nodes)
            if self.root.op_un_bin == 0:  # unary
                idx = np.random.choice(np.arange(self.offset, self.offset + self.num_nodes))
                self.adj[-1, idx] = 1
                if np.random.uniform(0, 1) >= decay_prob:
                    self.decay[-1, idx] = DecaySchedule()
            else:  # binary
                if np.random.uniform(0, 1) >= 0.20:  # operand
                    idx1 = np.random.choice(np.arange(0, self.offset))
                else:  # hidden state node
                    idx1 = np.random.choice(np.arange(self.offset, self.offset + self.num_nodes))
                self.adj[-1, idx1] = 1
                if np.random.uniform(0, 1) <= decay_prob:
                    self.decay[-1, idx1] = DecaySchedule()

                if idx1 < self.offset:  # first arg is operand, second must be hidden state
                    idx2 = np.random.choice(np.arange(self.offset, self.offset + self.num_nodes))
                else:  # first arg is hidden state, second can be either
                    if np.random.uniform(0, 1) >= 0.20:  # operand
                        idx2 = np.random.choice(np.arange(0, self.offset))
                    else:  # hidden state node
                        idx2 = np.random.choice(np.concatenate((np.arange(self.offset, idx1),
                                                                np.arange(idx1 + 1, self.offset + self.num_nodes))))
                self.adj[-1, idx2] = 2
                if np.random.uniform(0, 1) <= decay_prob:
                    self.decay[-1, idx2] = DecaySchedule()

            self.type = np.random.choice([0, 1, 2])  # SGD, Momentum, NESTEROV

            self.set_active()
            active_adj = self.adj[self.active_nodes >= 1][:, self.active_nodes >= 1]
            g = nx.from_numpy_array(active_adj, create_using=nx.DiGraph)
            try:
                nx.find_cycle(g)
                continue
            except:
                pass
            self.set_node_priority()
            if not self.check_integrity():
                continue
            return

    def set_node_priority(self):
        # orders nodes by their placement in graph
        nodes = np.where(self.active_nodes == 1)[0]
        priority = {}
        for node in nodes:
            priority[node] = None
        index = 0
        while True:
            for node in nodes:
                idx = np.where(self.adj[node] != 0)[0]
                if len(idx) == 0:
                    if priority[node] is None:
                        priority[node] = index
                        index += 1
                elif len(idx) == 1:
                    if priority[idx[0]] is None:
                        pass
                    else:
                        if priority[node] is None:
                            priority[node] = index
                            index += 1
                else:
                    if priority[idx[0]] is None or priority[idx[1]] is None:
                        pass
                    else:
                        if priority[node] is None:
                            priority[node] = index
                            index += 1
            if len(np.where(np.asarray(list(priority.values())) == None)[0]) == 0:
                break
        self.node_priority = np.asarray(
            list({k: v for k, v in sorted(priority.items(), key=lambda item: item[1])}.keys()))

    def mutate(self, msgs):
        r = np.random.uniform(0, 1)
        idx = np.random.choice(np.where(self.active_nodes[self.offset:] >= 1)[0].tolist())
        if idx == self.num_nodes:  # root
            node = self.root
        else:
            node = self.nodes[idx]

        if r >= 0.60:  # change op
            if node.op_un_bin == 0:  # unary
                if node.un == -1:
                    node.un = np.random.choice(np.concatenate((np.arange(0, node.un), np.arange(node.un + 1, 24))))
                else:
                    if np.random.uniform(0, 1) <= 0.20:  # 20% unary
                        node.un = -1
                    else:
                        node.un = np.random.choice(np.concatenate((np.arange(0, node.un), np.arange(node.un + 1, 24))))
            else:
                node.bin = np.random.choice(np.concatenate((np.arange(0, node.bin), np.arange(node.bin + 1, 10))))  #  was 9
        elif r >= 0.30:  # change conn
            c = np.random.choice(np.where(self.adj[idx + self.offset] == 0)[0])
            if node.op_un_bin == 0:  # unary
                if node == self.root:
                    c = np.random.choice(np.where(self.adj[idx + self.offset] == 0)[0][20:-1])

                self.adj[idx + self.offset][self.adj[idx + self.offset] == 1] = 0
                self.adj[idx + self.offset][c] = 1
            else:
                if np.random.uniform(0, 1) <= 0.20:  # swap conn
                    idx1 = np.where(self.adj[idx + self.offset] == 1)[0][0]
                    idx2 = np.where(self.adj[idx + self.offset] == 2)[0][0]
                    self.adj[idx + self.offset][idx1] = 2
                    self.adj[idx + self.offset][idx2] = 1
                else:
                    if node == self.root:
                        cons = np.where(self.adj[idx + self.offset] != 0)[0]
                        if np.sum(cons >= 20) == 2:  # both conns are hidden states
                            if np.random.uniform(0, 1) < 0.5:  # change 1st conn
                                self.adj[idx + self.offset][self.adj[idx + self.offset] == 1] = 0
                                self.adj[idx + self.offset][c] = 1
                            else:  # change 2nd conn           COPIED FROM BELOW!!!!!!!!!!
                                self.adj[idx + self.offset][self.adj[idx + self.offset] == 2] = 0
                                self.adj[idx + self.offset][c] = 2
                        else:
                            operand = np.where(cons < 20)[0][0]
                            if np.random.uniform(0, 1) < 0.5:  # mutate first conn

                                if self.adj[idx + self.offset][cons[operand]] == 1:  # first con is operand
                                    self.adj[idx + self.offset][self.adj[idx + self.offset] == 1] = 0
                                    self.adj[idx + self.offset][c] = 1
                                else:  # second is hidden state
                                    c = np.random.choice(np.where(self.adj[idx + self.offset] == 0)[0][20:-1])
                                    self.adj[idx + self.offset][self.adj[idx + self.offset] == 1] = 0
                                    self.adj[idx + self.offset][c] = 1
                            else:  # mutate hidden state conn

                                if self.adj[idx + self.offset][cons[operand]] == 2:  # second con is operand
                                    self.adj[idx + self.offset][self.adj[idx + self.offset] == 2] = 0
                                    self.adj[idx + self.offset][c] = 2
                                else:  # second is hidden state
                                    c = np.random.choice(np.where(self.adj[idx + self.offset] == 0)[0][20:-1])
                                    self.adj[idx + self.offset][self.adj[idx + self.offset] == 2] = 0
                                    self.adj[idx + self.offset][c] = 2
                    else:
                        if np.random.uniform(0, 1) < 0.5:  # change 1st conn
                            self.adj[idx + self.offset][self.adj[idx + self.offset] == 1] = 0
                            self.adj[idx + self.offset][c] = 1
                        else:  # change 2nd conn
                            self.adj[idx + self.offset][self.adj[idx + self.offset] == 2] = 0
                            self.adj[idx + self.offset][c] = 2
        elif r >= 0.20:  # change un->bin visa versa
            c = np.random.choice(np.where(self.adj[idx + self.offset] == 0)[0])
            if node.op_un_bin == 0:  # unary -> binary
                node.op_un_bin = 1
                if np.random.uniform(0, 1) <= 0.5:  # add right conn
                    self.adj[idx + self.offset][c] = 2
                else:
                    self.adj[idx + self.offset][self.adj[idx + self.offset] == 1] = 2
                    self.adj[idx + self.offset][c] = 1
            else:  # binary -> unary
                node.op_un_bin = 0
                if node == self.root:  # ensure root node binary has hidden state connection
                    cons = np.where(self.adj[idx + self.offset] != 0)[0]
                    if np.sum(cons >= 20) == 2:
                        if np.random.uniform(0, 1) <= 0.5:  # delete right conn
                            self.adj[idx + self.offset][self.adj[idx + self.offset] == 2] = 0
                        else:
                            self.adj[idx + self.offset][self.adj[idx + self.offset] == 1] = 0
                            self.adj[idx + self.offset][self.adj[idx + self.offset] == 2] = 1
                    else:
                        operand = np.where(cons < 20)[0][0]
                        if self.adj[idx + self.offset][cons[operand]] == 1:  # first con is operand
                            self.adj[idx + self.offset][self.adj[idx + self.offset] == 1] = 0
                            self.adj[idx + self.offset][self.adj[idx + self.offset] == 2] = 1
                        else:
                            self.adj[idx + self.offset][self.adj[idx + self.offset] == 2] = 0
                else:
                    if np.random.uniform(0, 1) <= 0.5:  # delete right conn
                        self.adj[idx + self.offset][self.adj[idx + self.offset] == 2] = 0
                    else:
                        self.adj[idx + self.offset][self.adj[idx + self.offset] == 1] = 0
                        self.adj[idx + self.offset][self.adj[idx + self.offset] == 2] = 1
        elif r >= 0.10:
            self.type = np.random.choice(np.concatenate((np.arange(0, self.type), np.arange(self.type + 1, 3))))
        else:
            idx_decay = np.random.choice(np.where(self.adj[idx + self.offset] != 0)[0])
            if self.decay[idx + self.offset][idx_decay] is None:
                self.decay[idx + self.offset][idx_decay] = DecaySchedule()
            else:
                if np.random.uniform(0, 1) >= 0.50:  # delete with 50%
                    self.decay[idx + self.offset][idx_decay] = None
                else:
                    self.decay[idx + self.offset][idx_decay].mutate([self.decay[idx + self.offset][idx_decay].msg],
                                                                    [self.decay[idx + self.offset][
                                                                         idx_decay].phenotype])
        self.set_active()
        active_adj = self.adj[self.active_nodes >= 1][:, self.active_nodes >= 1]
        g = nx.from_numpy_array(active_adj, create_using=nx.DiGraph)
        try:
            nx.find_cycle(g)
            return False
        except:
            pass

        self.set_node_priority()
        if not self.check_integrity():
            return False

        if self.msg in msgs:
            return False

        return True

    def print_nodes(self):
        idx = np.where(self.adj[-1] >= 1)[0]
        if self.root.op_un_bin == 0:
            msg = "root: unary(node{})".format(idx[0] - self.offset)
        else:
            idx1 = np.where(self.adj[-1] == 1)[0][0]
            idx2 = np.where(self.adj[-1] == 2)[0][0]
            msg = "root: bin(node{}, node{})".format(idx1 - 4, idx2 - 4)
        for i in [3, 2, 1, 0]:
            msg = msg + "\n"
            idx = np.where(self.adj[4 + i] >= 1)[0]
            if self.nodes[i].op_un_bin == 0:  # unary
                if idx >= 4:
                    msg = msg + "node{}: unary(node{})".format(i, idx[0] - 4)
                else:
                    msg = msg + "node{}: unary({})".format(i, idx[0])
            else:  # binary
                idx1 = np.where(self.adj[4 + i] == 1)[0][0]
                idx2 = np.where(self.adj[4 + i] == 2)[0][0]
                if idx1 >= 4 and idx2 >= 4:
                    msg = msg + "node{}: binary(node{}, node{})".format(i, idx1 - 4, idx2 - 4)
                elif idx1 >= 4:
                    msg = msg + "node{}: binary(node{}, {})".format(i, idx1 - 4, idx2)
                elif idx2 >= 4:
                    msg = msg + "node{}: binary({}, node{})".format(i, idx1, idx2 - 4)
                else:
                    msg = msg + "node{}: binary({}, {})".format(i, idx1, idx2)
        return msg + "\n"

    def set_active(self):
        self.root.active = True
        queue = []
        visited = [self.num_nodes + self.offset]
        queue = np.concatenate((queue, np.where(self.adj[-1] != 0)[0].flatten())).tolist()
        self.active_nodes = np.zeros(shape=(self.offset + self.num_nodes + 1,))
        self.active_nodes[-1] = 1
        while queue:
            node = int(queue.pop(0))
            if node in visited:
                continue
            visited.append(node)
            queue = np.concatenate((queue, np.where(self.adj[node] != 0)[0].flatten())).tolist()
            self.active_nodes[node] = 1
        self.node_priority = np.asarray(visited)
        self.active_nodes = np.asarray(self.active_nodes, dtype=int)
        return

    def sphere(self, opt):
        r = [1.1, .9, 1.1, .9, 1]
        x = tf.Variable(r, trainable=True, dtype=tf.float32)
        b = [0.5, 1.5, -0.3, -1.2, 0.75]
        for i in range(0, 35):
            with tf.GradientTape() as grad:
                loss = tf.math.reduce_sum(tf.math.pow(x - b, 2))
            trainable_weights = [x]
            # print(lr, i, trainable_weights)
            gradients = grad.gradient(loss, trainable_weights)
            opt.apply_gradients(zip(gradients, trainable_weights))
        return loss.numpy()

    def degenerate_opt_detection(self):
        for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            opt = Optimizer(learning_rate=lr, nodes=self, total_steps=35, warmup_steps=6, static=False, use_decay=True)
            val = self.sphere(opt)
            if val < 3.5:
                return False, False, opt.msg

        for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
            opt = Optimizer(learning_rate=lr, nodes=self, total_steps=35, warmup_steps=6, static=False, flip=True,
                            use_decay=True)
            val = self.sphere(opt)
            if val < 3.5:
                return False, True, opt.msg
        return True, True, opt.msg

    def check_integrity(self):

        try:
            t1, t2, msg = self.degenerate_opt_detection()
        except Exception as err:
            print(err)
            pickle.dump(self, open("ERROR_{}".format(err), "wb"))
            print("EXCEPTION")
            return False
        if t1:
            return False
        if t2:
            self.flip = True
        self.msg = msg
        return True

    def set_nodes(self):
        pass




def fitness_function_test(ind):
    if np.random.uniform(0, 1) < 0.2:
        return None
    res = []
    for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        opt = Optimizer(learning_rate=lr, nodes=ind, total_steps=35, warmup_steps=6, static=False, flip=ind.flip,
                            use_decay=True)  # PowerSign(learning_rate=lr))
        res.append(ind.sphere(opt))
    return (np.nanmin(res)+1e-8)*np.random.normal(1, 0.001), None, None

loss = tf.keras.losses.CategoricalCrossentropy()

def fitness_function_standard_opt(opt):
    acc = []
    lrs = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    for lr in lrs:
        model = ConvNet("swish")
        scheduler = WarmupCosineDecay(total_steps=8000, warmup_steps=800, hold=0, target_lr=lr)
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
        history = model.fit(trainloader.repeat(), epochs=4, steps_per_epoch=200,
                            callbacks=[TerminateOnNaN(), scheduler],
                            verbose=1)
        acc.append(np.nanmax(history.history['accuracy']))
    lr = lrs[np.argmax(acc)]
    model = ConvNet("swish")
    scheduler = WarmupCosineDecay(total_steps=8000, warmup_steps=800, hold=0, target_lr=lr)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    history = model.fit(trainloader.repeat(), epochs=40, steps_per_epoch=200, validation_data=testloader,
                        callbacks=[TerminateOnNaN(), scheduler],
                        verbose=1)
    f = np.nanmax(history.history['val_accuracy'])
    v = np.nanmax(history.history['val_loss'])
    if np.nanmax(history.history['accuracy']) < threshold:
        return None
    return f, v, history.history


def fitness_function_init(ind):
    acc = []
    lrs = [10, 1e-2, 1e-5]
    for lr in lrs:
        model = ConvNet("swish")
        optimizer = Optimizer(learning_rate=lr, nodes=ind, flip=ind.flip, total_steps=8000, warmup_steps=800,
                              static=False, use_decay=True)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(trainloader.repeat(), epochs=4, steps_per_epoch=200,
                            callbacks=[TerminateOnNaN(), FiveEpochCallback(epoch=2, threshold=0.25)],
                            verbose=0)
        acc.append(np.nanmax(history.history['accuracy']))
    if np.all(np.asarray(acc) < 0.25):
        return None
    lrs = lrs + [1, 1e-1, 1e-3, 1e-4]
    for i in range(3, len(lrs)):
        lr = lrs[i]
        model = ConvNet("swish")
        optimizer = Optimizer(learning_rate=lr, nodes=ind, flip=ind.flip, total_steps=8000, warmup_steps=800,
                              static=False, use_decay=True)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(trainloader.repeat(), epochs=4, steps_per_epoch=200,
                            callbacks=[TerminateOnNaN(), FiveEpochCallback(epoch=2, threshold=0.25)],
                            verbose=0)
        acc.append(np.nanmax(history.history['accuracy']))
    if np.nanmax(acc) < threshold:
        return None
    lr = lrs[np.argmax(acc)]
    model = ConvNet("swish")
    optimizer = Optimizer(learning_rate=lr, nodes=ind, flip=ind.flip, total_steps=8000, warmup_steps=800,
                          static=False, use_decay=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(trainloader.repeat(), epochs=40, steps_per_epoch=200, validation_data=testloader,
                        callbacks=[TerminateOnNaN(), FiveEpochCallback(epoch=4, threshold=threshold)],
                        verbose=0)
    f = np.nanmax(history.history['val_accuracy'])
    v = np.nanmax(history.history['val_loss'])
    return f, v, [f, v]


(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
y_train_david = tf.keras.utils.to_categorical(y_train, 10)[train_ind]
y_test_david = tf.keras.utils.to_categorical(y_train, 10)[val_ind]

train_mean = np.mean(x_train, axis=(0, 1, 2))
train_std = np.std(x_train, axis=(0, 1, 2))

normalize = lambda x: ((x - train_mean) / train_std).astype('float32')
pad4 = lambda x: np.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)], mode='reflect')

x_train_david = normalize(pad4(x_train[train_ind]))
x_test_david = normalize(x_train[val_ind])
import tensorflow_addons as tfa

data_aug = lambda x, y: (
    tfa.image.random_cutout(tf.image.random_flip_left_right(tf.image.random_crop(x, [32, 32, 3]))[None, ...],
                            mask_size=(8, 8))[0], y)

AUTO = tf.data.experimental.AUTOTUNE

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_david, y_train_david)).map(data_aug,
                                                                                       num_parallel_calls=AUTO).batch(
    BATCH_SIZE).prefetch(AUTO)
test_set = tf.data.Dataset.from_tensor_slices((x_test_david, y_test_david)).batch(BATCH_SIZE)


def fitness_function_evolution(ind, base=48):
    acc = []
    lrs = [10, 1e-2, 1e-5]
    for lr in lrs:
        model = ConvNet("swish", base=base)
        optimizer = Optimizer(learning_rate=lr, nodes=ind, flip=ind.flip, total_steps=16000, warmup_steps=800,
                              static=False, use_decay=True)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(train_dataset.repeat(), epochs=4, steps_per_epoch=200,
                            callbacks=[TerminateOnNaN(), FiveEpochCallback(epoch=2, threshold=0.25)],
                            verbose=0)
        acc.append(np.nanmax(history.history['accuracy']))
    if np.all(np.asarray(acc) < 0.25):
        return None
    lrs = lrs + [1, 1e-1, 1e-3, 1e-4]
    for i in range(3, len(lrs)):
        lr = lrs[i]
        model = ConvNet("swish", base=base)
        optimizer = Optimizer(learning_rate=lr, nodes=ind, flip=ind.flip, total_steps=16000, warmup_steps=800,
                              static=False, use_decay=True)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(train_dataset.repeat(), epochs=4, steps_per_epoch=200,
                            callbacks=[TerminateOnNaN(), FiveEpochCallback(epoch=2, threshold=0.25)],
                            verbose=0)
        acc.append(np.nanmax(history.history['accuracy']))
    if np.nanmax(acc) < threshold:
        return None
    lr = lrs[np.argmax(acc)]
    model = ConvNet("swish", base=base)
    optimizer = Optimizer(learning_rate=lr, nodes=ind, flip=ind.flip, total_steps=16000, warmup_steps=800,
                          static=False, use_decay=True)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(train_dataset.repeat(), epochs=40, steps_per_epoch=400, validation_data=test_set,
                        callbacks=[TerminateOnNaN(), FiveEpochCallback(epoch=4, threshold=threshold)],
                        verbose=0)
    f = np.nanmax(history.history['val_accuracy'])
    v = np.nanmax(history.history['val_loss'])
    return f, v, history.history












import os

import copy


class ParticleMutationAlgorithm:

    def __init__(self, n, k, m):
        self.gen_size = n
        self.gen = []
        self.fitness = []
        self.prev_individuals = []
        self.prev_fitness = []
        self.best_fit = []
        self.mean_fit = []
        self.median_fit = []
        self.min_fit = []
        self.similarities = []
        self.phenotypes = []
        self.functions = []
        self.index = self.gen_size
        self.K = k
        self.M = m
        self.init_gen = []
        self.init_fitness = []

    def initialize(self, fitness_function, init_size):
        """
        Purpose:
           - Create an initial population of randomly sampled Optimizers and evaluate on cheap ConvNet

        Loop over init_size iterations
           - create an OptimizerChrom()
           - evaluate OptimizerChrom through fitness_function
           - add Optimizer if passed the fitness_function

        """

        msg = "TRAINING INITIAL POPULATION"
        print(msg)
        logging.info(msg)
        self.init_gen = []
        self.init_fitness = []
        for i in range(0, init_size):
            s1 = time.time()
            self.init_gen.append(OptimizerChrom())  # random opt
            result = fitness_function(self.init_gen[i])
            if result is None:
                msg = " INIT MODEL ARCHITECTURE FAILED..."
                self.init_fitness.append(0)
            else:
                f, v, hist = result
                self.init_gen[i].hist = hist
                f1 = time.time()
                msg = " MODEL {}/{} -> Val Acc: {}, Val Loss: {}, Time: {}, Fun: {}".format(i, init_size, f, v, f1 - s1,
                                                                                         self.init_gen[i].msg)
                self.init_fitness.append(f)
            print(msg)
            logging.info(msg)

        # Only take the best to form the initial particles
        self.init_fitness = np.asarray(self.init_fitness)
        self.init_gen = np.asarray(self.init_gen)
        bst = np.argsort(-self.init_fitness)[0:self.gen_size]
        self.fitness = self.init_fitness[bst]
        self.gen = self.init_gen[bst]

    def evolve(self, fitness_function, START=0):
        """
        Purpose:
          - Read Paper

        For each Initial Particle
           - Re-evaluate on new fitness function
           - Loop over K TimeSteps per particle
              - Loop over M times, mutating particle, evaluating
              - Take the best from mutation and use as initial particle for next timestep
        """

        start = time.time()
        for i in range(START, self.gen_size):
            particle = self.gen[i]
            s = time.time()
            result = fitness_function(particle)
            track = []
            fitness = []
            if result is None:
                msg = "INIT PARTICLE {}/{} : {} : FAILED".format(i+1, self.gen_size, particle.msg)
                f = -1
            else:
                f, v, hist = result
                msg = "INIT PARTICLE {}/{} : {} : val: {} - Time: {}".format(i+1, self.gen_size, particle.msg, f, time.time()-s)
            print(msg)
            logging.info(msg)
            fitness.append(f)
            track.append(particle)
            msgs = [particle.msg]
            for k in range(0, self.K):
                msg = " Depth: {}/{}".format(k + 1, self.K)
                print(msg)
                logging.info(msg)
                vs = []
                ps = []
                for m in range(0, self.M):
                    print("  Mutation: {}/{}".format(m + 1, self.M))
                    REDO = 4
                    s = time.time()
                    while REDO > 0:
                        redo = True
                        while redo:
                            p1 = copy.deepcopy(particle)
                            if p1.mutate(msgs):
                                redo = False
                        result = fitness_function(p1)
                        if result is None:
                            msg = "    {} : FAILED".format(p1.msg)
                            f = -1
                            REDO -= 1
                        else:
                            f, v, hist = result
                            REDO = - 1
                            msg = "    {} val: {} - Time: {}".format(p1.msg, f, time.time() - s)
                        print(msg)
                        logging.info(msg)
                        msgs.append(p1.msg)
                    vs.append(f)
                    ps.append(p1)
                    track.append(p1)
                    fitness.append(f)
                bst = np.argmax(vs)
                particle = ps[bst]
                v = vs[bst]
                msg = "  Chosen {} : {} - ACCUMULATIVE TIME: {}\n".format(particle.msg, v, time.time() - start)
                print(msg)
                logging.info(msg)

            self.prev_individuals.append(track)
            self.prev_fitness.append(fitness)
            msg = " --- STATE SAVE ITER {} --- \n".format(i+1)
            print(msg)
            logging.info(msg)
            pickle.dump(self, open(args.save_dir + "/" + args.save_file + "_STATE_SAVE_ITER_{}".format(i), "wb"))

        finish = time.time()

        msg = " --- Time Elapsed: {} min --- ".format((finish - start) / 60.0)
        print(msg)
        logging.info(msg)



def create_parser():
    '''
    Create a command line parser for the XOR experiment
    '''
    parser = argparse.ArgumentParser(description='Neural Loss Evolution')
    parser.add_argument('--logs_file', type=str, default='nas_optimizer_particle_n50_k5_m7.log',
                        help='Output File For Logging')
    parser.add_argument('--save_dir', type=str, default='nas_optimizer_particle_n50_k5_m7',
                        help='Save Directory for saving Logs/Checkups')
    parser.add_argument('--save_file', type=str, default='nas_optimizer_particle_n50_k5_m7',
                        help='Save File for Algorithm')

    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(filename=args.logs_file, level=logging.DEBUG)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    n = 50  # number of particles
    k = 5  # number of mutations per iteration
    m = 7  # number of iterations
    algo = ParticleMutationAlgorithm(n=n, k=k, m=m)

    # initial population
    algo.initialize(fitness_function_init, init_size=10 * n)

    # save the file
    pickle.dump(algo, open(args.save_dir + "/" + args.save_file + "_RANDOM_INIT_n{}_k{}_m{}".format(n, k, m), "wb"))
    start = time.time()
    msg = "--- Starting Evolution ---"
    logging.info(msg)
    print(msg)

    algo.evolve(fitness_function_evolution)
    pickle.dump(algo, open(args.save_dir + "/" + args.save_file + "_FINAL_n{}_k{}_m{}".format(n, k, m), "wb"))
    finish = time.time()

    msg = "--- ENDING Evolution ---"
    print(msg)
    logging.info(msg)

    msg = "--- Total Time Taken: {} min ---".format((finish - start) / 60.0)
    logging.info(msg)
    print(msg)

