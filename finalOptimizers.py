import numpy as np
import math
import tensorflow as tf

import keras.backend as K




from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer


class EvolvedLRSchedules(tf.keras.callbacks.Callback):
    def __init__(self, total_steps=0, warmup_steps=0, start_lr=0.0, target_lr=1e-3, hold=0, type=0):
        super(EvolvedLRSchedules, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.global_step = 0
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.lrs = []
        self.lr = tf.convert_to_tensor(
                     tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32))

        deriv_tanh = lambda x: 1-tf.math.pow(tf.math.tanh(x), 2)
        deriv_soft = lambda x: tf.math.divide(1, 1 + tf.math.abs(x))

        t = np.arange(0, self.total_steps)
        a1 = 1.0 - tf.math.divide(t, self.total_steps)
        a2 = tf.math.divide(t, self.total_steps)
        a3 = 0.5 * (1 + tf.math.cos(tf.math.divide(math.pi * t, self.total_steps)))
        a4 = 0.5 * (1 - tf.math.cos(tf.math.divide(math.pi * t, self.total_steps)))
        a5 = 0.5 * (1 + tf.math.cos(tf.math.divide(2 * math.pi * t, self.total_steps)))
        a6 = 0.5 * (1 - tf.math.cos(tf.math.divide(2 * math.pi * t, self.total_steps)))
        a7 = 0.5 * (1 + tf.math.cos(math.pi * tf.math.divide(np.mod(2 * t, self.total_steps), self.total_steps)))
        a8 = 0.5 * (1 - tf.math.cos(math.pi * tf.math.divide(np.mod(2 * t, self.total_steps), self.total_steps)))
        a9 = 1 - tf.math.divide(np.mod(2 * t, self.total_steps), self.total_steps)
        a10 = tf.math.divide(np.mod(2 * t, self.total_steps), self.total_steps)
        a11 = tf.math.pow(0.01, tf.math.divide(t, self.total_steps))
        a12 = 1 - tf.math.pow(0.01, tf.math.divide(t, self.total_steps))
        beta_init = 0.95
        decay_rate = 1 - tf.math.divide((t + 1), self.total_steps)
        beta_decay = beta_init * decay_rate
        a13 = tf.math.divide(beta_decay, ((1.0 - beta_init) + beta_decay))
        a14 = tf.math.divide(beta_init - beta_decay, ((1.0 - beta_init) + beta_decay))

        self.type = type
        lrs = [
            self.lr * tf.cast(tf.math.erfc(tf.math.erfc(a4)) / deriv_tanh(a8), dtype=tf.float32),
            self.lr * tf.cast(tf.math.erfc(tf.math.erfc(a4)) / (deriv_tanh(a8)*deriv_tanh(a4)), dtype=tf.float32),
            self.lr * tf.cast(tf.math.atan(a2) / (deriv_tanh(a10) * tf.math.sqrt(deriv_soft(a14))), dtype=tf.float32),
            self.lr * tf.cast(tf.math.pow(tf.math.sigmoid(a2), 2) / tf.math.sigmoid(2 * tf.math.softsign(a1)), dtype=tf.float32),
            self.lr / tf.cast(tf.math.sqrt(tf.math.erf(a11)), dtype=tf.float32),
            self.lr * tf.cast(tf.math.atan(a2) * tf.math.erfc(a6) / tf.math.sqrt(deriv_soft(a6)), dtype=tf.float32),
            self.lr / tf.cast(tf.math.sqrt(deriv_soft(a10)), dtype=tf.float32),
            self.lr / (tf.cast(tf.math.sqrt(tf.math.softsign(tf.math.atan(a12))), dtype=tf.float32)+1e-7),
            self.lr * tf.cast(tf.math.tanh(tf.math.maximum(a10, a6)), dtype=tf.float32)
        ]
        self.lr = lrs[self.type]
        self.lr[0] = 1e-7

    def lr_warmup_cosine_decay(self, t):
        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (
                1 + tf.math.cos(
            math.pi * (t - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        warmup_lr = self.target_lr * (t / self.warmup_steps)

        if self.hold > 0:
            learning_rate = tf.where(t > self.warmup_steps + self.hold,
                                     learning_rate, self.target_lr)

        learning_rate = tf.where(t < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def call(self):
        lr = self.lr[self.global_step]
        return lr

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1

    def on_batch_begin(self, batch, logs=None):
        lr = self.lr[self.global_step]
        K.set_value(self.model.optimizer.lr, lr + 1e-7)
        #tf.print(self.model.optimizer.lr)


class AdamClip(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, use_locking=False, name="CustOpt", flip=False,
                 total_steps=0, warmup_steps=0, start_lr=0.0, hold=0, static=True, use_decay=False, variant=0):
        super(AdamClip, self).__init__(use_locking, name)
        self._flip = flip
        self._iterations = 0
        self._msg = None
        self.use_decay = use_decay
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.target_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.variant = variant
        self.static = static
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.lr = tf.convert_to_tensor(
            tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32))
        self.variant = variant

    def _create_slots(self, var_list):
        for var in var_list:
            self._zeros_slot(var, "v", self._name)
            self._zeros_slot(var, "s", self._name)

    def _finish(self, update_ops, name_scope):
        t = self.t.assign_add(1.0)
        return control_flow_ops.group(*update_ops + [t], name=name_scope)

    def lr_warmup_cosine_decay(self, t):
        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (
                1 + tf.math.cos(
            math.pi * (t - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        warmup_lr = self.target_lr * (t / self.warmup_steps)

        if self.hold > 0:
            learning_rate = tf.where(t > self.warmup_steps + self.hold,
                                     learning_rate, self.target_lr)

        learning_rate = tf.where(t < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self.lr, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]  # self._lr_t
        b9 = tf.cast(0.9, var.dtype.base_dtype)
        b99 = tf.cast(0.99, var.dtype.base_dtype)
        one = tf.cast(1, var.dtype.base_dtype)

        v = self.get_slot(var, "v")
        v.assign(b9 * v + (one - b9) * grad)
        s = self.get_slot(var, "s")
        s.assign(b99 * s + (one - b99) * grad * grad)

        if self.variant == 0:
            s1 = tf.math.sqrt(tf.math.divide(s, one - tf.math.pow(b99, self.t + 1))) + 1e-7
            update = tf.clip_by_value(tf.math.divide(v, one - tf.math.pow(b9, self.t + 1)),
                                      clip_value_min=-s1, clip_value_max=s1)
        elif self.variant == 1:
            s1 = tf.math.abs(tf.math.log(tf.math.divide(s, one - tf.math.pow(b99, self.t + 1)) + 1e-7))
            update = tf.clip_by_value(tf.math.divide(v, one - tf.math.pow(b9, self.t + 1)),
                                      clip_value_min=-s1, clip_value_max=s1)
        elif self.variant == 2:
            s1 = tf.math.sqrt(tf.math.abs(tf.math.log(tf.math.divide(s, one - tf.math.pow(b99, self.t + 1)) + 1e-7)))
            update = tf.clip_by_value(tf.math.divide(v, one - tf.math.pow(b9, self.t + 1)),
                                      clip_value_min=-s1, clip_value_max=s1)
        elif self.variant == 3:
            s1 = tf.math.sigmoid(tf.math.divide(s, one - tf.math.pow(b99, self.t + 1)) + 1e-7)
            update = tf.clip_by_value(tf.math.divide(v, one - tf.math.pow(b9, self.t + 1)),
                                      clip_value_min=-s1, clip_value_max=s1)
        elif self.variant == 4:
            s1 = tf.math.sqrt(tf.math.divide(s, one - tf.math.pow(b99, self.t + 1))) + 1e-7
            update = tf.clip_by_norm(tf.clip_by_value(tf.math.divide(v, one - tf.math.pow(b9, self.t + 1)),
                                                      clip_value_min=-s1, clip_value_max=s1), 1)
        else:
            print("ERROR HERE")
        var_update = state_ops.assign_add(var, - lr_t * update)

        return control_flow_ops.group(*[var_update, v, s])


class Opt1(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, use_locking=False, name="CustOpt", flip=False,
                 total_steps=0, warmup_steps=0, start_lr=0.0, hold=0, static=True, use_decay=False, variant=0):
        super(Opt1, self).__init__(use_locking, name)
        self._flip = flip
        self._iterations = 0
        self._msg = None
        self.use_decay = use_decay
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.target_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.variant = variant
        self.static = static
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.lr = tf.convert_to_tensor(
            tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32))
        self.variant = variant

    def _create_slots(self, var_list):
        for var in var_list:
            self._zeros_slot(var, "v", self._name)
            self._zeros_slot(var, "s", self._name)

    def _finish(self, update_ops, name_scope):
        t = self.t.assign_add(1.0)
        return control_flow_ops.group(*update_ops + [t], name=name_scope)

    def lr_warmup_cosine_decay(self, t):
        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (
                1 + tf.math.cos(
            math.pi * (t - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        warmup_lr = self.target_lr * (t / self.warmup_steps)

        if self.hold > 0:
            learning_rate = tf.where(t > self.warmup_steps + self.hold,
                                     learning_rate, self.target_lr)

        learning_rate = tf.where(t < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self.lr, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]  # self._lr_t
        b9 = tf.cast(0.9, var.dtype.base_dtype)
        b99 = tf.cast(0.99, var.dtype.base_dtype)
        one = tf.cast(1, var.dtype.base_dtype)
        b7 = tf.cast(0.70, var.dtype.base_dtype)
        b95 = tf.cast(0.95, var.dtype.base_dtype)
        v = self.get_slot(var, "v")
        v.assign(b9 * v + (one - b9) * grad)
        s = self.get_slot(var, "s")
        s.assign(b99 * s + (one - b99) * grad * grad)

        s1 = tf.math.abs((1e-5 * var) - ((one - b95) * tf.math.pow(grad, 2)) + b95 * tf.math.divide(s, one - tf.math.pow(b99, self.t + 1)) )
        update = (one - b7) * grad + b7 * tf.math.divide(v, one - tf.math.pow(b9, self.t + 1)) + tf.math.softsign(tf.clip_by_value(1e-5*var, clip_value_min=-s1, clip_value_max=s1))

        var_update = state_ops.assign_add(var, - lr_t * update)

        return control_flow_ops.group(*[var_update, v, s])


class Opt2(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, use_locking=False, name="CustOpt", flip=False,
                 total_steps=0, warmup_steps=0, start_lr=0.0, hold=0, static=True, use_decay=False, variant=0):
        super(Opt2, self).__init__(use_locking, name)
        self._flip = flip
        self._iterations = 0
        self._msg = None
        self.use_decay = use_decay
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.target_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.variant = variant
        self.static = static
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.lr = tf.convert_to_tensor(
            tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32))
        self.variant = variant

    def _create_slots(self, var_list):
        for var in var_list:
            self._zeros_slot(var, "v", self._name)
            self._zeros_slot(var, "s", self._name)

    def _finish(self, update_ops, name_scope):
        t = self.t.assign_add(1.0)
        return control_flow_ops.group(*update_ops + [t], name=name_scope)

    def lr_warmup_cosine_decay(self, t):
        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (
                1 + tf.math.cos(
            math.pi * (t - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        warmup_lr = self.target_lr * (t / self.warmup_steps)

        if self.hold > 0:
            learning_rate = tf.where(t > self.warmup_steps + self.hold,
                                     learning_rate, self.target_lr)

        learning_rate = tf.where(t < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self.lr, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]  # self._lr_t
        b9 = tf.cast(0.9, var.dtype.base_dtype)
        b99 = tf.cast(0.99, var.dtype.base_dtype)
        one = tf.cast(1, var.dtype.base_dtype)
        b7 = tf.cast(0.70, var.dtype.base_dtype)
        b95 = tf.cast(0.95, var.dtype.base_dtype)
        v = self.get_slot(var, "v")
        v.assign(b9 * v + (one - b9) * grad)
        s = self.get_slot(var, "s")
        s.assign(b99 * s + (one - b99) * grad * grad)

        s1 = 1e-5 * var - ((one - b95) * tf.math.pow(grad, 2) + b95 * tf.math.divide(s, one - tf.math.pow(b99, self.t + 1)))
        update = (one - b7) * grad + b7 * tf.math.divide(v, one - tf.math.pow(b9, self.t + 1)) + tf.math.softsign(tf.math.divide(1e-5 * var, tf.math.sqrt(1 + tf.math.pow(s1, 2))))

        var_update = state_ops.assign_add(var, - lr_t * update)

        return control_flow_ops.group(*[var_update, v, s])


class Opt3(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, use_locking=False, name="CustOpt", flip=False,
                 total_steps=0, warmup_steps=0, start_lr=0.0, hold=0, static=True, use_decay=False, variant=0):
        super(Opt3, self).__init__(use_locking, name)
        self._flip = flip
        self._iterations = 0
        self._msg = None
        self.use_decay = use_decay
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.target_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.variant = variant
        self.static = static
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.lr = tf.convert_to_tensor(
            tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32))
        self.variant = variant

    def _create_slots(self, var_list):
        for var in var_list:
            self._zeros_slot(var, "v", self._name)
            self._zeros_slot(var, "l", self._name)

    def _finish(self, update_ops, name_scope):
        t = self.t.assign_add(1.0)
        return control_flow_ops.group(*update_ops + [t], name=name_scope)

    def lr_warmup_cosine_decay(self, t):
        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (
                1 + tf.math.cos(
            math.pi * (t - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        warmup_lr = self.target_lr * (t / self.warmup_steps)

        if self.hold > 0:
            learning_rate = tf.where(t > self.warmup_steps + self.hold,
                                     learning_rate, self.target_lr)

        learning_rate = tf.where(t < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self.lr, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]  # self._lr_t
        b9 = tf.cast(0.9, var.dtype.base_dtype)
        b99 = tf.cast(0.99, var.dtype.base_dtype)
        one = tf.cast(1, var.dtype.base_dtype)
        b7 = tf.cast(0.70, var.dtype.base_dtype)
        b999 = tf.cast(0.999, var.dtype.base_dtype)
        v = self.get_slot(var, "v")
        v.assign(b9 * v + (one - b9) * grad)
        l = self.get_slot(var, "l")
        l.assign(b999 * l + (one - b999) * grad * grad)

        l1 = 1e-5 * var - (
                    (one - b99) * tf.math.pow(grad, 3) + b99 * tf.math.divide(l, one - tf.math.pow(b999, self.t + 1)))

        update = (one - b7) * grad + b7 * tf.math.divide(v, one - tf.math.pow(b9, self.t + 1)) + tf.math.softsign(
            tf.math.divide(1e-5 * var, tf.math.sqrt(1 + tf.math.pow(l1, 2))))

        var_update = state_ops.assign_add(var, - lr_t * update)

        return control_flow_ops.group(*[var_update, v, l])


class Opt4(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, use_locking=False, name="CustOpt", flip=False,
                 total_steps=0, warmup_steps=0, start_lr=0.0, hold=0, static=True, use_decay=False, variant=0):
        super(Opt4, self).__init__(use_locking, name)
        self._flip = flip
        self._iterations = 0
        self._msg = None
        self.use_decay = use_decay
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.target_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.static = static
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        t = tf.cast(np.arange(0, self.total_steps), dtype=tf.float32)

        a4 = 0.5 * (1 - tf.math.cos(tf.math.divide(math.pi * t, self.total_steps)))

        a8 = 0.5 * (1 - tf.math.cos(math.pi * tf.math.divide(np.mod(2 * t, self.total_steps), self.total_steps)))

        beta_init = 0.95
        decay_rate = 1 - tf.math.divide((t + 1), self.total_steps)

        beta_decay = beta_init * decay_rate
        a13 = tf.math.divide(beta_decay, ((1.0 - beta_init) + beta_decay))
        self.a13 = tf.cast(tf.math.atan(a13), dtype=tf.float32)
        self.variant = variant
        if self.use_decay:
            lr = tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32) * tf.math.erfc(tf.math.erfc(a4)) / tf.cast(1-tf.math.pow(tf.math.tanh(a8), 2.0), dtype=tf.float32)
        else:
            print("NOT USING DECAY!!!")
            lr = tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32)
        self.lr = tf.convert_to_tensor(
            tf.cast(lr, dtype=tf.float32))


    def _create_slots(self, var_list):
        for var in var_list:
            self._zeros_slot(var, "v", self._name)

    def _finish(self, update_ops, name_scope):
        t = self.t.assign_add(1.0)
        return control_flow_ops.group(*update_ops + [t], name=name_scope)

    def lr_warmup_cosine_decay(self, t):
        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (
                1 + tf.math.cos(
            math.pi * (t - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        warmup_lr = self.target_lr * (t / self.warmup_steps)

        if self.hold > 0:
            learning_rate = tf.where(t > self.warmup_steps + self.hold,
                                     learning_rate, self.target_lr)

        learning_rate = tf.where(t < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self.lr, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]  # self._lr_t
        if self.variant == 1:
            a13 = tf.cast(1.0, var.dtype.base_dtype)
        else:
            a13 = tf.cast(self.a13, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]
        b9 = tf.cast(0.9, var.dtype.base_dtype)
        one = tf.cast(1, var.dtype.base_dtype)
        b7 = tf.cast(0.70, var.dtype.base_dtype)
        v = self.get_slot(var, "v")
        v.assign(b9 * v + (one - b9) * grad)

        c = tf.math.abs(a13*tf.math.exp(tf.math.divide(v, one - tf.math.pow(b9, self.t + 1))))
        update = tf.math.divide((one - b7) * grad + b7 * tf.math.divide(v, one - tf.math.pow(b9, self.t + 1)),
                       tf.clip_by_value(2 * tf.ones_like(var), clip_value_min=-c, clip_value_max=c))

        var_update = state_ops.assign_add(var, - lr_t * update)

        return control_flow_ops.group(*[var_update, v])


class Opt5(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, use_locking=False, name="CustOpt", flip=False,
                 total_steps=0, warmup_steps=0, start_lr=0.0, hold=0, static=True, use_decay=False):
        super(Opt5, self).__init__(use_locking, name)
        self._flip = flip
        self._iterations = 0
        self._msg = None
        self.use_decay = use_decay
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.target_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.static = static
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        t = tf.cast(np.arange(0, self.total_steps), dtype=tf.float32)

        a4 = 0.5 * (1 - tf.math.cos(tf.math.divide(math.pi * t, self.total_steps)))

        a8 = 0.5 * (1 - tf.math.cos(math.pi * tf.math.divide(np.mod(2 * t, self.total_steps), self.total_steps)))

        beta_init = 0.95
        decay_rate = 1 - tf.math.divide((t + 1), self.total_steps)

        beta_decay = beta_init * decay_rate
        a13 = tf.math.divide(beta_decay, ((1.0 - beta_init) + beta_decay))
        self.a13 = tf.cast(tf.math.atan(a13), dtype=tf.float32)
        if self.use_decay:
            lr = tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32) * tf.math.erfc(tf.math.erfc(a4)) / tf.cast(1-tf.math.pow(tf.math.tanh(a8), 2.0), dtype=tf.float32)
        else:
            print("NOT USING DECAY!!!")
            lr = tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32)
        self.lr = tf.convert_to_tensor(
            tf.cast(lr, dtype=tf.float32))

    def _create_slots(self, var_list):
        for var in var_list:
            self._zeros_slot(var, "v", self._name)

    def _finish(self, update_ops, name_scope):
        t = self.t.assign_add(1.0)
        return control_flow_ops.group(*update_ops + [t], name=name_scope)

    def lr_warmup_cosine_decay(self, t):
        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (
                1 + tf.math.cos(
            math.pi * (t - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        warmup_lr = self.target_lr * (t / self.warmup_steps)

        if self.hold > 0:
            learning_rate = tf.where(t > self.warmup_steps + self.hold,
                                     learning_rate, self.target_lr)

        learning_rate = tf.where(t < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self.lr, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]  # self._lr_t
        b9 = tf.cast(0.9, var.dtype.base_dtype)
        one = tf.cast(1, var.dtype.base_dtype)
        b7 = tf.cast(0.70, var.dtype.base_dtype)
        v = self.get_slot(var, "v")
        v.assign(b9 * v + (one - b9) * grad)

        c = tf.math.abs(tf.math.exp(tf.math.divide(v, one - tf.math.pow(b9, self.t + 1))))
        update = tf.math.divide((one - b7) * grad + b7 * tf.math.divide(v, one - tf.math.pow(b9, self.t + 1)),
                       tf.clip_by_value(2 * tf.ones_like(var), clip_value_min=-c, clip_value_max=c))

        var_update = state_ops.assign_add(var, - lr_t * update)

        return control_flow_ops.group(*[var_update, v])


class Opt6(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, use_locking=False, name="CustOpt", flip=False,
                 total_steps=0, warmup_steps=0, start_lr=0.0, hold=0, static=True, use_decay=False):
        super(Opt6, self).__init__(use_locking, name)
        self._flip = flip
        self._iterations = 0
        self._msg = None
        self.use_decay = use_decay
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.target_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.static = static
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        t = tf.cast(np.arange(0, self.total_steps), dtype=tf.float32)

        a4 = 0.5 * (1 - tf.math.cos(tf.math.divide(math.pi * t, self.total_steps)))

        a8 = 0.5 * (1 - tf.math.cos(math.pi * tf.math.divide(np.mod(2 * t, self.total_steps), self.total_steps)))

        self.a4 = tf.cast((1 - tf.math.pow(tf.math.tanh(a4), 2)), dtype=tf.float32)
        if self.use_decay:
            lr = tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32) * tf.math.erfc(
                tf.math.erfc(a4)) / tf.cast(1-tf.math.pow(tf.math.tanh(a8), 2.0), dtype=tf.float32)
        else:
            print("NOT USING DECAY!!!")
            lr = tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32)
        self.lr = tf.convert_to_tensor(
            tf.cast(lr, dtype=tf.float32))

    def _create_slots(self, var_list):
        for var in var_list:
            self._zeros_slot(var, "v", self._name)

    def _finish(self, update_ops, name_scope):
        t = self.t.assign_add(1.0)
        return control_flow_ops.group(*update_ops + [t], name=name_scope)

    def lr_warmup_cosine_decay(self, t):
        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (
                1 + tf.math.cos(
            math.pi * (t - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        warmup_lr = self.target_lr * (t / self.warmup_steps)

        if self.hold > 0:
            learning_rate = tf.where(t > self.warmup_steps + self.hold,
                                     learning_rate, self.target_lr)

        learning_rate = tf.where(t < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self.lr, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]  # self._lr_t
        if self.use_decay:
            a4 = tf.cast(self.a4, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]
        else:
            a4 = tf.cast(1.0, var.dtype.base_dtype)
        b9 = tf.cast(0.9, var.dtype.base_dtype)
        one = tf.cast(1, var.dtype.base_dtype)
        b7 = tf.cast(0.70, var.dtype.base_dtype)
        v = self.get_slot(var, "v")
        v.assign(b9 * v + (one - b9) * grad)

        update = tf.math.divide((one - b7) * grad + b7 * tf.math.divide(v, one - tf.math.pow(b9, self.t + 1)),
                               tf.math.abs(a4 * tf.math.exp(1e-4 * var)))

        var_update = state_ops.assign_add(var, - lr_t * update)

        return control_flow_ops.group(*[var_update, v])


class Opt7(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, use_locking=False, name="CustOpt", flip=False,
                 total_steps=0, warmup_steps=0, start_lr=0.0, hold=0, static=True, use_decay=False):
        super(Opt7, self).__init__(use_locking, name)
        self._flip = flip
        self._iterations = 0
        self._msg = None
        self.use_decay = use_decay
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.target_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.static = static
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        t = tf.cast(np.arange(0, self.total_steps), dtype=tf.float32)

        a6 = tf.cast(0.5 * (1 - tf.math.cos(tf.math.divide(2 * math.pi * t, self.total_steps))), dtype=tf.float32)
        a10 = tf.cast(tf.math.divide(np.mod(2 * t, self.total_steps), self.total_steps), dtype=tf.float32)

        self.a = tf.math.maximum(a6, a10)
        lr = tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32)
        self.lr = tf.convert_to_tensor(
            tf.cast(lr, dtype=tf.float32))

    def _create_slots(self, var_list):
        for var in var_list:
            self._zeros_slot(var, "v", self._name)

    def _finish(self, update_ops, name_scope):
        t = self.t.assign_add(1.0)
        return control_flow_ops.group(*update_ops + [t], name=name_scope)

    def lr_warmup_cosine_decay(self, t):
        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (
                1 + tf.math.cos(
            math.pi * (t - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        warmup_lr = self.target_lr * (t / self.warmup_steps)

        if self.hold > 0:
            learning_rate = tf.where(t > self.warmup_steps + self.hold,
                                     learning_rate, self.target_lr)

        learning_rate = tf.where(t < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self.lr, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]  # self._lr_t
        if self.use_decay:
            a = tf.cast(self.a, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]
        else:
            a = tf.cast(1.0, var.dtype.base_dtype)
        b9 = tf.cast(0.9, var.dtype.base_dtype)
        one = tf.cast(1, var.dtype.base_dtype)
        b7 = tf.cast(0.70, var.dtype.base_dtype)
        v = self.get_slot(var, "v")
        v.assign(b9 * v + (one - b9) * grad)

        update = tf.math.tanh(a*tf.math.asinh((one - b7) * grad + b7 * tf.math.divide(v, one - tf.math.pow(b9, self.t + 1))))

        var_update = state_ops.assign_add(var, - lr_t * update)

        return control_flow_ops.group(*[var_update, v])


class Opt8(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, use_locking=False, name="CustOpt", flip=False,
                 total_steps=0, warmup_steps=0, start_lr=0.0, hold=0, static=True, use_decay=False):
        super(Opt8, self).__init__(use_locking, name)
        self._flip = flip
        self._iterations = 0
        self._msg = None
        self.use_decay = use_decay
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.target_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.static = static
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        t = tf.cast(np.arange(0, self.total_steps), dtype=tf.float32)

        a6 = tf.cast(0.5 * (1 - tf.math.cos(tf.math.divide(2 * math.pi * t, self.total_steps))), dtype=tf.float32)
        a10 = tf.cast(tf.math.divide(np.mod(2 * t, self.total_steps), self.total_steps), dtype=tf.float32)

        self.a = tf.math.maximum(a6, a10)
        lr = tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32)
        self.lr = tf.convert_to_tensor(
            tf.cast(lr, dtype=tf.float32))

    def _create_slots(self, var_list):
        for var in var_list:
            self._zeros_slot(var, "v", self._name)

    def _finish(self, update_ops, name_scope):
        t = self.t.assign_add(1.0)
        return control_flow_ops.group(*update_ops + [t], name=name_scope)

    def lr_warmup_cosine_decay(self, t):
        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (
                1 + tf.math.cos(
            math.pi * (t - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        warmup_lr = self.target_lr * (t / self.warmup_steps)

        if self.hold > 0:
            learning_rate = tf.where(t > self.warmup_steps + self.hold,
                                     learning_rate, self.target_lr)

        learning_rate = tf.where(t < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self.lr, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]  # self._lr_t
        if self.use_decay:
            a = tf.cast(self.a, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]
        else:
            a = tf.cast(1.0, var.dtype.base_dtype)
        b9 = tf.cast(0.9, var.dtype.base_dtype)
        one = tf.cast(1, var.dtype.base_dtype)
        b7 = tf.cast(0.70, var.dtype.base_dtype)
        v = self.get_slot(var, "v")
        v.assign(b9 * v + (one - b9) * grad)

        update = tf.math.asinh(a*tf.math.asinh((one - b7) * grad + b7 * tf.math.divide(v, one - tf.math.pow(b9, self.t + 1))))

        var_update = state_ops.assign_add(var, - lr_t * update)

        return control_flow_ops.group(*[var_update, v])


class Opt9(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, use_locking=False, name="CustOpt", flip=False,
                 total_steps=0, warmup_steps=0, start_lr=0.0, hold=0, static=True, use_decay=False, variant=0):
        super(Opt9, self).__init__(use_locking, name)
        self._flip = flip
        self._iterations = 0
        self._msg = None
        self.use_decay = use_decay
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.target_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.variant = variant
        print("HERE I AM DIFFERNT")
        self.static = static
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        t = tf.cast(np.arange(0, self.total_steps), dtype=tf.float32)
        a11 = tf.math.pow(0.01, tf.math.divide(t, self.total_steps))
        self.a = tf.math.erfc(a11)
        self.lr = tf.convert_to_tensor(
            tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32))
        t = np.concatenate((np.linspace(0.95, 0.85, self.warmup_steps),
                            np.linspace(0.85, 0.85, self.hold),
                            np.linspace(0.85, 0.95, self.total_steps - self.hold - self.warmup_steps)))
        self.m = tf.convert_to_tensor(tf.cast(t, dtype=tf.float32))
        self.variant = variant

    def _create_slots(self, var_list):
        for var in var_list:
            self._zeros_slot(var, "s", self._name)
            self._zeros_slot(var, "mom", self._name)

    def _finish(self, update_ops, name_scope):
        t = self.t.assign_add(1.0)
        return control_flow_ops.group(*update_ops + [t], name=name_scope)

    def lr_warmup_cosine_decay(self, t):
        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (
                1 + tf.math.cos(
            math.pi * (t - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        warmup_lr = self.target_lr * (t / self.warmup_steps)

        if self.hold > 0:
            learning_rate = tf.where(t > self.warmup_steps + self.hold,
                                     learning_rate, self.target_lr)

        learning_rate = tf.where(t < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self.lr, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]  # self._lr_t
        if self.use_decay:
            a = tf.cast(self.a, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]
        else:
            a = tf.cast(1.0, var.dtype.base_dtype)
        m_t = tf.cast(self.m, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]
        b99 = tf.cast(0.99, var.dtype.base_dtype)
        one = tf.cast(1, var.dtype.base_dtype)
        b95 = tf.cast(0.95, var.dtype.base_dtype)
        s = self.get_slot(var, "s")
        s.assign(b99 * s + (one - b99) * grad * grad)

        update = a * grad * tf.math.exp(tf.math.atan((one - b95) * tf.math.pow(grad, 2) + b95 * tf.math.divide(s, one - tf.math.pow(b99, self.t + 1))))
        mom = self.get_slot(var, "mom")
        mom.assign(m_t * mom - lr_t * update)
        var_update = state_ops.assign_add(var, m_t * mom - lr_t * update)

        return control_flow_ops.group(*[var_update, s, mom])


class Opt10(optimizer.Optimizer):
    def __init__(self, learning_rate=0.001, use_locking=False, name="CustOpt", flip=False,
                 total_steps=0, warmup_steps=0, start_lr=0.0, hold=0, static=True, use_decay=False):
        super(Opt10, self).__init__(use_locking, name)
        self._flip = flip
        self._iterations = 0
        self._msg = None
        self.use_decay = use_decay
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        if not self.use_decay:
            print("HERE I AM DIFFERNT")
        self.target_lr = learning_rate
        self.warmup_steps = warmup_steps
        self.static = static
        self.t = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        t = tf.cast(np.arange(0, self.total_steps), dtype=tf.float32)
        a2 = tf.math.divide(t, self.total_steps)
        beta_init = 0.95
        decay_rate = 1 - tf.math.divide((t + 1), self.total_steps)
        beta_decay = beta_init * decay_rate
        a13 = tf.math.divide(beta_decay, ((1.0 - beta_init) + beta_decay))
        self.a = a13*a2
        self.lr = tf.convert_to_tensor(
            tf.cast(self.lr_warmup_cosine_decay(np.arange(0, self.total_steps)), dtype=tf.float32))
        t = np.concatenate((np.linspace(0.95, 0.85, self.warmup_steps),
                            np.linspace(0.85, 0.85, self.hold),
                            np.linspace(0.85, 0.95, self.total_steps - self.hold - self.warmup_steps)))
        self.m = tf.convert_to_tensor(tf.cast(t, dtype=tf.float32))

    def _create_slots(self, var_list):
        for var in var_list:
            self._zeros_slot(var, "mom", self._name)

    def _finish(self, update_ops, name_scope):
        t = self.t.assign_add(1.0)
        return control_flow_ops.group(*update_ops + [t], name=name_scope)

    def lr_warmup_cosine_decay(self, t):
        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (
                1 + tf.math.cos(
            math.pi * (t - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        warmup_lr = self.target_lr * (t / self.warmup_steps)

        if self.hold > 0:
            learning_rate = tf.where(t > self.warmup_steps + self.hold,
                                     learning_rate, self.target_lr)

        learning_rate = tf.where(t < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def _resource_apply_dense(self, grad, var):
        lr_t = tf.cast(self.lr, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]  # self._lr_t
        if self.use_decay:
            a = tf.cast(self.a, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]
        else:
            a = tf.cast(1.0, var.dtype.base_dtype)
        m_t = tf.cast(self.m, var.dtype.base_dtype)[tf.cast(self.t, tf.int64)]

        update = tf.math.bessel_i1e(tf.math.bessel_i1e(a * grad))
        mom = self.get_slot(var, "mom")
        mom.assign(m_t * mom - lr_t * update)
        var_update = state_ops.assign_add(var, m_t * mom - lr_t * update)

        return control_flow_ops.group(*[var_update, mom])

# HOW TO CREATE OPTIMIZER AND USE IT IN MODEL
# -------------------------------------------
# lr = 1.0 # whatever you want the max to be
# optimizer = Opt1(total_steps=96000, warmup_steps=6400, hold=12800, learning_rate=lr)
# or
# optimizer = AdamClip(total_steps=96000, warmup_steps=6400, hold=12800, learning_rate=lr, variant=1)
# model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

# What about EvolvedLR Schedules?
# -------------------------------------------
# schedule = EvolvedLRSchedules(total_steps=96000, warmup_steps=6400, hold=12800, type=1)
# model.fit(callbacks=[schedule], ...)