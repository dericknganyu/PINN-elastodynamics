import numpy as np
import pickle

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Setup GPU for training
import tensorflow.compat.v1 as tf # type: ignore
tf.disable_v2_behavior()
import tensorflow as tf1
# import deepxde as dde

import os
import time

import wandb # type: ignore
wandb.require("core")

from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(1111)
tf.set_random_seed(1111)


# tf.logging.set_verbosity(tf.logging.ERROR)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class PINN:
    # Initialize the class
    def __init__(self, Collo, HOLE, IC, LF, RT, UP, LW, DIST, uv_layers, dist_layers, part_layers, lb, ub, direct,
                 partDir='', distDir='', uvDir='', run_num = 0):

        self.direct = direct
        # Count for callback function
        self.count = 0
        self.it = 0
        self.t0 = 0
        self.t1 = 0
        self.run_num = run_num
        # Bounds
        self.lb = lb
        self.ub = ub

        # Mat. properties
        self.E = 20.0
        self.mu = 0.25
        self.rho = 1.0
        self.hole_r = 0.1

        # Collocation point
        self.x_c = Collo[:, 0:1]
        self.y_c = Collo[:, 1:2]
        self.t_c = Collo[:, 2:3]

        # Source wave
        self.x_HOLE = HOLE[:, 0:1]
        self.y_HOLE = HOLE[:, 1:2]
        self.t_HOLE = HOLE[:, 2:3]

        # Initial condition point, t=0
        self.x_IC = IC[:, 0:1]
        self.y_IC = IC[:, 1:2]
        self.t_IC = IC[:, 2:3]

        # Symmetry bound, where u=s12=0
        self.x_LF = LF[:, 0:1]
        self.y_LF = LF[:, 1:2]
        self.t_LF = LF[:, 2:3]

        # Loading bound, where s12=0, s11=func(t)
        self.x_RT = RT[:, 0:1]
        self.y_RT = RT[:, 1:2]
        self.t_RT = RT[:, 2:3]
        self.s11_RT = RT[:, 3:4]

        # Free bound, where s12=s22=0
        self.x_UP = UP[:, 0:1]
        self.y_UP = UP[:, 1:2]
        self.t_UP = UP[:, 2:3]

        # Symmetry bound, where s12=0, v=0
        self.x_LW = LW[:, 0:1]
        self.y_LW = LW[:, 1:2]
        self.t_LW = LW[:, 2:3]

        # Distance function
        self.x_dist = DIST[:, 0:1]
        self.y_dist = DIST[:, 1:2]
        self.t_dist = DIST[:, 2:3]
        self.u_dist = DIST[:, 3:4]
        self.v_dist = DIST[:, 4:5]
        self.s11_dist = DIST[:, 5:6]
        self.s22_dist = DIST[:, 6:7]
        self.s12_dist = DIST[:, 7:8]

        # Define layers config
        self.uv_layers = uv_layers
        self.dist_layers = dist_layers
        self.part_layers = part_layers

        # Load trained network if provided, else initialize them randomly.
        if distDir == '':
            self.dist_weights, self.dist_biases = self.initialize_NN(self.dist_layers)
        else:
            print("Loading dist NN ...")
            self.dist_weights, self.dist_biases = self.load_NN(direct+distDir, self.dist_layers)

        if partDir == '':
            self.part_weights, self.part_biases = self.initialize_NN(self.part_layers)
        else:
            print("Loading part NN ...")
            self.part_weights, self.part_biases = self.load_NN(direct+partDir, self.part_layers)

        if uvDir=='':
            self.uv_weights, self.uv_biases = self.initialize_NN(self.uv_layers)
        else:
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN(direct+uvDir, self.uv_layers)

        # tf placeholders
        self.learning_rate = tf.placeholder(tf.float64, shape=[])
        self.x_tf = tf.placeholder(tf.float64, shape=[None, self.x_c.shape[1]])  # Point for postprocessing
        self.y_tf = tf.placeholder(tf.float64, shape=[None, self.y_c.shape[1]])
        self.t_tf = tf.placeholder(tf.float64, shape=[None, self.t_c.shape[1]])

        self.x_c_tf = tf.placeholder(tf.float64, shape=[None, self.x_c.shape[1]])
        self.y_c_tf = tf.placeholder(tf.float64, shape=[None, self.y_c.shape[1]])
        self.t_c_tf = tf.placeholder(tf.float64, shape=[None, self.t_c.shape[1]])

        self.x_HOLE_tf = tf.placeholder(tf.float64, shape=[None, self.x_HOLE.shape[1]])
        self.y_HOLE_tf = tf.placeholder(tf.float64, shape=[None, self.y_HOLE.shape[1]])
        self.t_HOLE_tf = tf.placeholder(tf.float64, shape=[None, self.t_HOLE.shape[1]])

        self.x_IC_tf = tf.placeholder(tf.float64, shape=[None, self.x_IC.shape[1]])
        self.y_IC_tf = tf.placeholder(tf.float64, shape=[None, self.y_IC.shape[1]])
        self.t_IC_tf = tf.placeholder(tf.float64, shape=[None, self.t_IC.shape[1]])

        self.x_LF_tf = tf.placeholder(tf.float64, shape=[None, self.x_LF.shape[1]])
        self.y_LF_tf = tf.placeholder(tf.float64, shape=[None, self.y_LF.shape[1]])
        self.t_LF_tf = tf.placeholder(tf.float64, shape=[None, self.t_LF.shape[1]])

        self.x_RT_tf = tf.placeholder(tf.float64, shape=[None, self.x_RT.shape[1]])
        self.y_RT_tf = tf.placeholder(tf.float64, shape=[None, self.y_RT.shape[1]])
        self.t_RT_tf = tf.placeholder(tf.float64, shape=[None, self.t_RT.shape[1]])
        self.s11_RT_tf = tf.placeholder(tf.float64, shape=[None, self.s11_RT.shape[1]])

        self.x_UP_tf = tf.placeholder(tf.float64, shape=[None, self.x_UP.shape[1]])
        self.y_UP_tf = tf.placeholder(tf.float64, shape=[None, self.y_UP.shape[1]])
        self.t_UP_tf = tf.placeholder(tf.float64, shape=[None, self.t_UP.shape[1]])

        self.x_LW_tf = tf.placeholder(tf.float64, shape=[None, self.x_LW.shape[1]])
        self.y_LW_tf = tf.placeholder(tf.float64, shape=[None, self.y_LW.shape[1]])
        self.t_LW_tf = tf.placeholder(tf.float64, shape=[None, self.t_LW.shape[1]])

        self.x_dist_tf = tf.placeholder(tf.float64, shape=[None, self.x_dist.shape[1]])
        self.y_dist_tf = tf.placeholder(tf.float64, shape=[None, self.y_dist.shape[1]])
        self.t_dist_tf = tf.placeholder(tf.float64, shape=[None, self.t_dist.shape[1]])
        self.u_dist_tf = tf.placeholder(tf.float64, shape=[None, self.u_dist.shape[1]])
        self.v_dist_tf = tf.placeholder(tf.float64, shape=[None, self.v_dist.shape[1]])
        self.s11_dist_tf = tf.placeholder(tf.float64, shape=[None, self.s11_dist.shape[1]])
        self.s22_dist_tf = tf.placeholder(tf.float64, shape=[None, self.s22_dist.shape[1]])
        self.s12_dist_tf = tf.placeholder(tf.float64, shape=[None, self.s12_dist.shape[1]])

        # tf graphs
        self.u_pred, self.v_pred, self.s11_pred, self.s22_pred, self.s12_pred = self.net_uv(self.x_tf, self.y_tf, self.t_tf)
        self.e11_pred, self.e22_pred, self.e12_pred = self.net_e(self.x_tf, self.y_tf, self.t_tf)
        # Distance function
        self.D_u_pred, self.D_v_pred, self.D_s11_pred, self.D_s22_pred, self.D_s12_pred \
              = self.net_dist(self.x_dist_tf, self.y_dist_tf, self.t_dist_tf)

        # Time derivatives of distance func (to enforce zero-velocity IC later)
        self.dt_D_u_pred, self.dt_D_v_pred, self.dt_D_s11_pred, self.dt_D_s22_pred, self.dt_D_s12_pred \
              = self.net_dist_dt(self.x_IC_tf, self.y_IC_tf, self.t_IC_tf)
        # Particular solution (at boundary, initial state)
        self.P_u_pred, self.P_v_pred, self.P_s11_pred, self.P_s22_pred, self.P_s12_pred, _, _ \
              = self.net_part(self.x_tf, self.y_tf, self.t_tf)

        self.P_u_IC_pred, self.P_v_IC_pred, self.P_s11_IC_pred, self.P_s22_IC_pred, self.P_s12_IC_pred, \
              self.dt_P_u_IC_pred, self.dt_P_v_IC_pred = self.net_part(self.x_IC_tf, self.y_IC_tf, self.t_IC_tf)
        self.P_u_LF_pred, _, _, _, self.P_s12_LF_pred, _, _ = self.net_part(self.x_LF_tf, self.y_LF_tf, self.t_LF_tf)
        _, _, self.P_s11_RT_pred, _, self.P_s12_RT_pred, _, _ = self.net_part(self.x_RT_tf, self.y_RT_tf, self.t_RT_tf)
        _, self.P_v_LW_pred, _, _, self.P_s12_LW_pred, _, _ = self.net_part(self.x_LW_tf, self.y_LW_tf, self.t_LW_tf)
        _, _, _, self.P_s22_UP_pred, self.P_s12_UP_pred, _, _ = self.net_part(self.x_UP_tf, self.y_UP_tf, self.t_UP_tf)

        # Surface traction on circular hole
        self.tx_pred, self.ty_pred = self.net_t(self.x_HOLE_tf, self.y_HOLE_tf, self.t_HOLE_tf)
        # Governing eqn residual on collocation points
        self.f_pred_u, self.f_pred_v, self.f_pred_s11, self.f_pred_s22, self.f_pred_s12 = self.net_f_sig(self.x_c_tf,
                                                                                                         self.y_c_tf,
                                                                                                         self.t_c_tf)

        # Construct loss to optimize
        self.loss_f_uv = tf.reduce_mean(tf.square(self.f_pred_u)) \
                         + tf.reduce_mean(tf.square(self.f_pred_v))
        self.loss_f_s = tf.reduce_mean(tf.square(self.f_pred_s11)) \
                        + tf.reduce_mean(tf.square(self.f_pred_s22)) \
                        + tf.reduce_mean(tf.square(self.f_pred_s12))
        self.loss_HOLE = tf.reduce_mean(tf.square(self.tx_pred)) \
                         + tf.reduce_mean(tf.square(self.ty_pred))
        self.loss_DIST = tf.reduce_mean(tf.square(self.D_u_pred - self.u_dist_tf)) \
                         + tf.reduce_mean(tf.square(self.D_v_pred - self.v_dist_tf)) \
                         + tf.reduce_mean(tf.square(self.D_s11_pred - self.s11_dist_tf)) \
                         + tf.reduce_mean(tf.square(self.D_s22_pred - self.s22_dist_tf)) \
                         + tf.reduce_mean(tf.square(self.D_s12_pred - self.s12_dist_tf)) \
                         + tf.reduce_mean(tf.square(self.dt_D_u_pred)) \
                         + tf.reduce_mean(tf.square(self.dt_D_v_pred))
        self.loss_PART = tf.reduce_mean(tf.square(self.P_u_IC_pred)) \
                         + tf.reduce_mean(tf.square(self.P_v_IC_pred)) \
                         + tf.reduce_mean(tf.square(self.P_s11_IC_pred)) \
                         + tf.reduce_mean(tf.square(self.P_s22_IC_pred)) \
                         + tf.reduce_mean(tf.square(self.P_s12_IC_pred)) \
                         + tf.reduce_mean(tf.square(self.dt_P_u_IC_pred)) \
                         + tf.reduce_mean(tf.square(self.dt_P_v_IC_pred)) \
                         + tf.reduce_mean(tf.square(self.P_u_LF_pred)) \
                         + tf.reduce_mean(tf.square(self.P_s12_LF_pred)) \
                         + tf.reduce_mean(tf.square(self.P_s11_RT_pred - self.s11_RT_tf)) \
                         + tf.reduce_mean(tf.square(self.P_s12_RT_pred)) \
                         + tf.reduce_mean(tf.square(self.P_v_LW_pred)) \
                         + tf.reduce_mean(tf.square(self.P_s12_LW_pred)) \
                         + tf.reduce_mean(tf.square(self.P_s22_UP_pred)) \
                         + tf.reduce_mean(tf.square(self.P_s12_UP_pred)) \

        self.loss = 10 * (self.loss_f_uv + self.loss_f_s + self.loss_HOLE)

        # Optimizer to pretrain distance func network
        self.optimizer_dist = tf1.contrib.opt.ScipyOptimizerInterface(1000 * self.loss_DIST,
        # self.optimizer_dist = dde.optimizers.tensorflow_compat_v1.scipy_optimizer.ScipyOptimizerInterface(1000 * self.loss_DIST,
                                                                     var_list=self.dist_weights + self.dist_biases,
                                                                     method='L-BFGS-B',
                                                                     options={'maxiter': 20000,
                                                                              'maxfun': 20000,
                                                                              'maxcor': 50,
                                                                              'maxls': 50,
                                                                              'ftol': 0.00001 * np.finfo(float).eps})

        # Optimizer to pretrain particular solution network
        self.optimizer_part = tf1.contrib.opt.ScipyOptimizerInterface(1000 * self.loss_PART,
        # self.optimizer_part = dde.optimizers.tensorflow_compat_v1.scipy_optimizer.ScipyOptimizerInterface(1000 * self.loss_PART,
                                                                     var_list=self.part_weights + self.part_biases,
                                                                     method='L-BFGS-B',
                                                                     options={'maxiter': 20000,
                                                                              'maxfun': 20000,
                                                                              'maxcor': 50,
                                                                              'maxls': 50,
                                                                              'ftol': 0.00001 * np.finfo(float).eps})

        # Optimizer for final solution (while the dist, particular network freezed)
        self.optimizer = tf1.contrib.opt.ScipyOptimizerInterface(self.loss,
        # self.optimizer = dde.optimizers.tensorflow_compat_v1.scipy_optimizer.ScipyOptimizerInterface(self.loss,
                                                                var_list=self.uv_weights + self.uv_biases,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 70000,#150000,#70000,
                                                                         'maxfun': 70000,#150000,#70000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 0.00001 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss, var_list=self.uv_weights + self.uv_biases)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True)
        )#gpu_options = tf.GPUOptions(allow_growth=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float64),
                           dtype=tf.float64)

    def save_NN(self, fileDir, TYPE=''):
        if TYPE == 'UV':
            uv_weights = self.sess.run(self.uv_weights)
            uv_biases = self.sess.run(self.uv_biases)
        elif TYPE == 'DIST':
            uv_weights = self.sess.run(self.dist_weights)
            uv_biases = self.sess.run(self.dist_biases)
        elif TYPE == 'PART':
            uv_weights = self.sess.run(self.part_weights)
            uv_biases = self.sess.run(self.part_biases)
        else:
            pass
        with open(fileDir, 'wb') as f:
            pickle.dump([uv_weights, uv_biases], f)
            # print("Save " + TYPE + " NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        with open(fileDir, 'rb') as f:
            uv_weights, uv_biases = pickle.load(f)
            # Stored model must has the same # of layers
            assert num_layers == (len(uv_weights) + 1)
            for num in range(0, num_layers - 1):
                W = tf.Variable(uv_weights[num], dtype=tf.float64)
                b = tf.Variable(uv_biases[num], dtype=tf.float64)
                weights.append(W)
                biases.append(b)
                print("Load NN parameters successfully...")
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        # Note: for cases where X is very large or small, use the below normalization
        # H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_dist(self, x, y, t):
        dist = self.neural_net(tf.concat([x, y, t], 1), self.dist_weights, self.dist_biases)
        D_u = dist[:, 0:1]
        D_v = dist[:, 1:2]
        D_s11 = dist[:, 2:3]
        D_s22 = dist[:, 3:4]
        D_s12 = dist[:, 4:5]
        return D_u, D_v, D_s11, D_s22, D_s12

    def net_dist_dt(self, x, y, t):
        # Time derivative of distance function
        dist = self.neural_net(tf.concat([x, y, t], 1), self.dist_weights, self.dist_biases)
        D_u = dist[:, 0:1]
        D_v = dist[:, 1:2]
        D_s11 = dist[:, 2:3]
        D_s22 = dist[:, 3:4]
        D_s12 = dist[:, 4:5]

        dt_D_u = tf.gradients(D_u, t)[0]
        dt_D_v = tf.gradients(D_v, t)[0]
        dt_D_s11 = tf.gradients(D_s11, t)[0]
        dt_D_s22 = tf.gradients(D_s22, t)[0]
        dt_D_s12 = tf.gradients(D_s12, t)[0]
        return dt_D_u, dt_D_v, dt_D_s11, dt_D_s22, dt_D_s12

    def net_part(self, x, y, t):
        P = self.neural_net(tf.concat([x, y, t], 1), self.part_weights, self.part_biases)
        P_u = P[:, 0:1]
        P_v = P[:, 1:2]
        P_s11 = P[:, 2:3]
        P_s22 = P[:, 3:4]
        P_s12 = P[:, 4:5]
        dt_P_u = tf.gradients(P_u, t)[0]
        dt_P_v = tf.gradients(P_v, t)[0]
        return P_u, P_v, P_s11, P_s22, P_s12, dt_P_u, dt_P_v

    def net_uv(self, x, y, t):
        # Output for composite network, i.e. [u, v, s11, s22, s12]
        uv_sig = self.neural_net(tf.concat([x, y, t], 1), self.uv_weights, self.uv_biases)
        dist = self.neural_net(tf.concat([x, y, t], 1), self.dist_weights, self.dist_biases)
        part = self.neural_net(tf.concat([x, y, t], 1), self.part_weights, self.part_biases)

        u = uv_sig[:, 0:1]
        v = uv_sig[:, 1:2]
        s11 = uv_sig[:, 2:3]
        s22 = uv_sig[:, 3:4]
        s12 = uv_sig[:, 4:5]

        P_u = part[:, 0:1]
        P_v = part[:, 1:2]
        P_s11 = part[:, 2:3]
        P_s22 = part[:, 3:4]
        P_s12 = part[:, 4:5]

        D_u = dist[:, 0:1]
        D_v = dist[:, 1:2]
        D_s11 = dist[:, 2:3]
        D_s22 = dist[:, 3:4]
        D_s12 = dist[:, 4:5]

        #####p(x,y)+D(x,y)*u(x,y)######
        u = P_u + D_u * u
        v = P_v + D_v * v
        s11 = P_s11 + D_s11 * s11
        s22 = P_s22 + D_s22 * s22
        s12 = P_s12 + D_s12 * s12
        return u, v, s11, s22, s12

    def net_e(self, x, y, t):
        u, v, _, _, _ = self.net_uv(x, y, t)
        # Strains
        e11 = tf.gradients(u, x)[0]
        e22 = tf.gradients(v, y)[0]
        e12 = (tf.gradients(u, y)[0] + tf.gradients(v, x)[0])
        return e11, e22, e12

    def net_vel(self, x, y, t):
        u, v, _, _, _ = self.net_uv(x, y, t)
        u_t = tf.gradients(u, t)[0]
        v_t = tf.gradients(v, t)[0]
        return u_t, v_t

    def net_f_sig(self, x, y, t):
        # Network for governing (and constitutive) equation loss
        E = self.E
        mu = self.mu
        rho = self.rho

        u, v, s11, s22, s12 = self.net_uv(x, y, t)

        # Strains
        e11, e22, e12 = self.net_e(x, y, t)

        # Plane stress from auto-differentiation
        sp11 = E / (1 - mu * mu) * e11 + E * mu / (1 - mu * mu) * e22
        sp22 = E * mu / (1 - mu * mu) * e11 + E / (1 - mu * mu) * e22
        sp12 = E / (2 * (1 + mu)) * e12

        # raw stress output - stress from auto-differentiation
        f_s11 = s11 - sp11
        f_s12 = s12 - sp12
        f_s22 = s22 - sp22

        s11_1 = tf.gradients(s11, x)[0]
        s12_2 = tf.gradients(s12, y)[0]
        u_t = tf.gradients(u, t)[0]
        u_tt = tf.gradients(u_t, t)[0]

        s22_2 = tf.gradients(s22, y)[0]
        s12_1 = tf.gradients(s12, x)[0]
        v_t = tf.gradients(v, t)[0]
        v_tt = tf.gradients(v_t, t)[0]

        # f_u:=Sxx_x+Sxy_y-rho*u_tt
        f_u = s11_1 + s12_2 - rho * u_tt
        f_v = s22_2 + s12_1 - rho * v_tt

        return f_u, f_v, f_s11, f_s22, f_s12

    def net_surf_var(self, x, y, t, nx, ny):
        # In this function, the nx, ny for one edge is same
        # Return surface traction tx, ty

        u, v, s11, s22, s12 = self.net_uv(x, y, t)

        tx = tf.multiply(s11, nx) + tf.multiply(s12, ny)
        ty = tf.multiply(s12, nx) + tf.multiply(s22, ny)

        return tx, ty

    def net_t(self, x, y, t):
        # Calculate traction tx, ty for circular surface from coordinate x, y
        r = self.hole_r
        u, v, s11, s22, s12 = self.net_uv(x, y, t)
        # Todo: add nx, ny vector for each x, y
        nx = -x / r
        ny = -y / r
        tx = tf.multiply(s11, nx, name=None) + tf.multiply(s12, ny, name=None)
        ty = tf.multiply(s12, nx, name=None) + tf.multiply(s22, ny, name=None)
        return tx, ty

    def callback(self, loss, loss_f_uv, loss_f_s, loss_HOLE, save_fun):
        self.count = self.count + 1
        self.t1 = time.time()
        if self.count % 10 == 0:
            print('Epoch %s, Loss: %.6e in %0.3f s'%(self.count, loss, self.t1-self.t0))
        wandb.log({"gen/loss_f_uv": loss_f_uv,
                "gen/loss_f_s": loss_f_s,
                "gen/loss_HOLE": loss_HOLE,
                "gen/loss": loss
                })  
        if self.count % 1000 == 0:
                save_fun('%s/uvNN_float64.pickle_%s'%(self.direct, self.it + self.count + self.run_num), TYPE='UV')
                delete_files_if_exceeding_threshold(self.direct, 'uvNN_float64', threshold= 10)
        self.t0 = time.time()

    def callback_dist(self, loss_dist, Du_dist, Dv_dist, D11_dist, D22_dist, D12_dist, Dudt_dist, Dvdt_dist):
        self.count = self.count + 1
        self.t1 = time.time()
        if self.count % 10 == 0:
            print('Epoch %s, Loss: %.6e in %0.3f s'%(self.count, loss_dist, self.t1-self.t0))
        wandb.log({"dist/loss": loss_dist,
                "dist/Du": Du_dist,
                "dist/Dv": Dv_dist,
                "dist/D11": D11_dist,
                "dist/D22": D22_dist,
                "dist/D12": D12_dist,
                "dist/Dudt": Dudt_dist,
                "dist/Dvdt": Dvdt_dist,
                })  
        self.t0 = time.time() 

    def callback_part(self, loss_part, ICu_part, ICv_part, IC11_part, IC22_part, IC12_part, ICudt_part, ICvdt_part, 
                      LFu_part, LF12_part, RT11_part, RT12_part, LWv_part, LW12_part, UP22_part, UP12_part):
        self.count = self.count + 1
        self.t1 = time.time()
        if self.count % 10 == 0:
            print('Epoch %s, Loss: %.6e in %0.3f s'%(self.count, loss_part, self.t1-self.t0))
        wandb.log({"part/loss": loss_part,
                "part/ICu": ICu_part,
                "part/ICv": ICv_part,
                "part/IC11": IC11_part,
                "part/IC22": IC22_part,
                "part/IC12": IC12_part,
                "part/ICudt": ICudt_part,
                "part/ICvdt": ICvdt_part,
                "part/LFu": LFu_part,
                "part/LF12": LF12_part,
                "part/RT11": RT11_part,
                "part/RT12": RT12_part,
                "part/LWv": LWv_part,
                "part/LW12": LW12_part,
                "part/UP22": UP22_part,
                "part/UP12": UP12_part,
                })
        self.t0 = time.time()  

    def train(self, iter, learning_rate):

        loss_f_uv = []
        loss_f_s = []
        loss_HOLE = []
        loss = []

        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.t_c_tf: self.t_c,
                   self.x_IC_tf: self.x_IC, self.y_IC_tf: self.y_IC, self.t_IC_tf: self.t_IC,
                   self.x_HOLE_tf: self.x_HOLE, self.y_HOLE_tf: self.y_HOLE, self.t_HOLE_tf: self.t_HOLE,
                   self.x_LF_tf: self.x_LF, self.y_LF_tf: self.y_LF, self.t_LF_tf: self.t_LF,
                   self.x_RT_tf: self.x_RT, self.y_RT_tf: self.y_RT, self.t_RT_tf: self.t_RT,
                   self.s11_RT_tf: self.s11_RT,
                   self.x_UP_tf: self.x_UP, self.y_UP_tf: self.y_UP, self.t_UP_tf: self.t_UP,
                   self.x_LW_tf: self.x_LW, self.y_LW_tf: self.y_LW, self.t_LW_tf: self.t_LW,
                   self.x_dist_tf: self.x_dist, self.y_dist_tf: self.y_dist, self.t_dist_tf: self.t_dist,
                   self.u_dist_tf: self.u_dist,
                   self.v_dist_tf: self.v_dist, self.s11_dist_tf: self.s11_dist, self.s22_dist_tf: self.s22_dist,
                   self.s12_dist_tf: self.s12_dist,
                   self.learning_rate: learning_rate}

        for it in range(iter):
            self.sess.run(self.train_op_Adam, tf_dict)
            # Print
            if it % 10 == 0:
                self.t0 = time.time()
                loss_value = self.sess.run(self.loss, tf_dict)
                self.t1 = time.time()
                print('Epoch: %d, Loss: %.6e in %0.3f s' % (it, loss_value, self.t1-self.t0))
              
            if it % 1000 == 0:
                self.save_NN('%s/uvNN_float64.pickle_%s'%(self.direct, self.it + self.run_num), TYPE='UV')
                delete_files_if_exceeding_threshold(self.direct, 'uvNN_float64', threshold= 10)

            loss_f_uv.append(self.sess.run(self.loss_f_uv, tf_dict))
            loss_f_s.append(self.sess.run(self.loss_f_s, tf_dict))
            loss_HOLE.append(self.sess.run(self.loss_HOLE, tf_dict))
            loss.append(self.sess.run(self.loss, tf_dict))
            # Logging with WnB
            wandb.log({"gen/loss_f_uv": loss_f_uv[-1],
                    "gen/loss_f_s": loss_f_s[-1],
                    "gen/loss_HOLE": loss_HOLE[-1],
                    "gen/loss": loss[-1]
                    })  
            self.it = self.it + 1
        return loss_f_uv, loss_f_s, loss_HOLE, loss#, iter + offset + 1

    def train_bfgs(self):
        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.t_c_tf: self.t_c,
                   self.x_IC_tf: self.x_IC, self.y_IC_tf: self.y_IC, self.t_IC_tf: self.t_IC,
                   self.x_HOLE_tf: self.x_HOLE, self.y_HOLE_tf: self.y_HOLE, self.t_HOLE_tf: self.t_HOLE,
                   self.x_LF_tf: self.x_LF, self.y_LF_tf: self.y_LF, self.t_LF_tf: self.t_LF,
                   self.x_RT_tf: self.x_RT, self.y_RT_tf: self.y_RT, self.t_RT_tf: self.t_RT,
                   self.s11_RT_tf: self.s11_RT,
                   self.x_UP_tf: self.x_UP, self.y_UP_tf: self.y_UP, self.t_UP_tf: self.t_UP,
                   self.x_LW_tf: self.x_LW, self.y_LW_tf: self.y_LW, self.t_LW_tf: self.t_LW,
                   self.x_dist_tf: self.x_dist, self.y_dist_tf: self.y_dist, self.t_dist_tf: self.t_dist,
                   self.u_dist_tf: self.u_dist,
                   self.v_dist_tf: self.v_dist, self.s11_dist_tf: self.s11_dist, self.s22_dist_tf: self.s22_dist,
                   self.s12_dist_tf: self.s12_dist}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.loss_f_uv, self.loss_f_s, self.loss_HOLE, self.save_NN],
                                loss_callback=self.callback)

    def train_bfgs_dist(self):

        tf_dict = {self.x_IC_tf: self.x_IC, self.y_IC_tf: self.y_IC, self.t_IC_tf: self.t_IC,
                   self.x_HOLE_tf: self.x_HOLE, self.y_HOLE_tf: self.y_HOLE, self.t_HOLE_tf: self.t_HOLE,
                   self.x_LF_tf: self.x_LF, self.y_LF_tf: self.y_LF, self.t_LF_tf: self.t_LF,
                   self.x_RT_tf: self.x_RT, self.y_RT_tf: self.y_RT, self.t_RT_tf: self.t_RT,
                   self.s11_RT_tf: self.s11_RT,
                   self.x_UP_tf: self.x_UP, self.y_UP_tf: self.y_UP, self.t_UP_tf: self.t_UP,
                   self.x_LW_tf: self.x_LW, self.y_LW_tf: self.y_LW, self.t_LW_tf: self.t_LW,
                   self.x_dist_tf: self.x_dist, self.y_dist_tf: self.y_dist, self.t_dist_tf: self.t_dist,
                   self.u_dist_tf: self.u_dist,
                   self.v_dist_tf: self.v_dist, self.s11_dist_tf: self.s11_dist, self.s22_dist_tf: self.s22_dist,
                   self.s12_dist_tf: self.s12_dist}

        self.optimizer_dist.minimize(self.sess,
                                     feed_dict=tf_dict,
                                     fetches=[self.loss_DIST,
                                              tf.reduce_mean(tf.square(self.D_u_pred - self.u_dist_tf)),
                                              tf.reduce_mean(tf.square(self.D_v_pred - self.v_dist_tf)),
                                              tf.reduce_mean(tf.square(self.D_s11_pred - self.s11_dist_tf)),
                                              tf.reduce_mean(tf.square(self.D_s22_pred - self.s22_dist_tf)),
                                              tf.reduce_mean(tf.square(self.D_s12_pred - self.s12_dist_tf)),
                                              tf.reduce_mean(tf.square(self.dt_D_u_pred)),
                                              tf.reduce_mean(tf.square(self.dt_D_v_pred))
                                              ],
                                     loss_callback=self.callback_dist)

    def train_bfgs_part(self):

        tf_dict = {self.x_IC_tf: self.x_IC, self.y_IC_tf: self.y_IC, self.t_IC_tf: self.t_IC,
                   self.x_HOLE_tf: self.x_HOLE, self.y_HOLE_tf: self.y_HOLE, self.t_HOLE_tf: self.t_HOLE,
                   self.x_LF_tf: self.x_LF, self.y_LF_tf: self.y_LF, self.t_LF_tf: self.t_LF,
                   self.x_RT_tf: self.x_RT, self.y_RT_tf: self.y_RT, self.t_RT_tf: self.t_RT,
                   self.s11_RT_tf: self.s11_RT,
                   self.x_UP_tf: self.x_UP, self.y_UP_tf: self.y_UP, self.t_UP_tf: self.t_UP,
                   self.x_LW_tf: self.x_LW, self.y_LW_tf: self.y_LW, self.t_LW_tf: self.t_LW}

        self.optimizer_part.minimize(self.sess,
                                     feed_dict=tf_dict,
                                     fetches=[self.loss_PART,
                                              tf.reduce_mean(tf.square(self.P_u_IC_pred)),
                                              tf.reduce_mean(tf.square(self.P_v_IC_pred)),
                                              tf.reduce_mean(tf.square(self.P_s11_IC_pred)),
                                              tf.reduce_mean(tf.square(self.P_s22_IC_pred)),
                                              tf.reduce_mean(tf.square(self.P_s12_IC_pred)),
                                              tf.reduce_mean(tf.square(self.dt_P_u_IC_pred)),
                                              tf.reduce_mean(tf.square(self.dt_P_v_IC_pred)),
                                              tf.reduce_mean(tf.square(self.P_u_LF_pred)),
                                              tf.reduce_mean(tf.square(self.P_s12_LF_pred)),
                                              tf.reduce_mean(tf.square(self.P_s11_RT_pred - self.s11_RT_tf)),
                                              tf.reduce_mean(tf.square(self.P_s12_RT_pred)),
                                              tf.reduce_mean(tf.square(self.P_v_LW_pred)),
                                              tf.reduce_mean(tf.square(self.P_s12_LW_pred)),
                                              tf.reduce_mean(tf.square(self.P_s22_UP_pred)),
                                              tf.reduce_mean(tf.square(self.P_s12_UP_pred))],
                                     loss_callback=self.callback_part)

    def predict(self, x_star, y_star, t_star):
        u_star = self.sess.run(self.u_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        v_star = self.sess.run(self.v_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        s11_star = self.sess.run(self.s11_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        s22_star = self.sess.run(self.s22_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        s12_star = self.sess.run(self.s12_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        e11_star = self.sess.run(self.e11_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        e22_star = self.sess.run(self.e22_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        e12_star = self.sess.run(self.e12_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        return u_star, v_star, s11_star, s22_star, s12_star, e11_star, e22_star, e12_star

    def predict_D(self, x_star, y_star, t_star):
        D_u = self.sess.run(self.D_u_pred, {self.x_dist_tf: x_star, self.y_dist_tf: y_star, self.t_dist_tf: t_star})
        D_v = self.sess.run(self.D_v_pred, {self.x_dist_tf: x_star, self.y_dist_tf: y_star, self.t_dist_tf: t_star})
        D_s11 = self.sess.run(self.D_s11_pred, {self.x_dist_tf: x_star, self.y_dist_tf: y_star, self.t_dist_tf: t_star})
        D_s22 = self.sess.run(self.D_s22_pred, {self.x_dist_tf: x_star, self.y_dist_tf: y_star, self.t_dist_tf: t_star})
        D_s12 = self.sess.run(self.D_s12_pred, {self.x_dist_tf: x_star, self.y_dist_tf: y_star, self.t_dist_tf: t_star})
        return D_u, D_v, D_s11, D_s22, D_s12

    def predict_P(self, x_star, y_star, t_star):
        P_u = self.sess.run(self.P_u_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        P_v = self.sess.run(self.P_v_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        P_s11 = self.sess.run(self.P_s11_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        P_s22 = self.sess.run(self.P_s22_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        P_s12 = self.sess.run(self.P_s12_pred, {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star})
        return P_u, P_v, P_s11, P_s22, P_s12

    def getloss(self):
        tf_dict = {self.x_c_tf: self.x_c, self.y_c_tf: self.y_c, self.t_c_tf: self.t_c,
                   self.x_IC_tf: self.x_IC, self.y_IC_tf: self.y_IC, self.t_IC_tf: self.t_IC,
                   self.x_HOLE_tf: self.x_HOLE, self.y_HOLE_tf: self.y_HOLE, self.t_HOLE_tf: self.t_HOLE,
                   self.x_LF_tf: self.x_LF, self.y_LF_tf: self.y_LF, self.t_LF_tf: self.t_LF,
                   self.x_RT_tf: self.x_RT, self.y_RT_tf: self.y_RT, self.t_RT_tf: self.t_RT,
                   self.s11_RT_tf: self.s11_RT,
                   self.x_UP_tf: self.x_UP, self.y_UP_tf: self.y_UP, self.t_UP_tf: self.t_UP,
                   self.x_LW_tf: self.x_LW, self.y_LW_tf: self.y_LW, self.t_LW_tf: self.t_LW,
                   self.x_dist_tf: self.x_dist, self.y_dist_tf: self.y_dist, self.t_dist_tf: self.t_dist,
                   self.u_dist_tf: self.u_dist,
                   self.v_dist_tf: self.v_dist, self.s11_dist_tf: self.s11_dist, self.s22_dist_tf: self.s22_dist,
                   self.s12_dist_tf: self.s12_dist}
        loss_f_uv = self.sess.run(self.loss_f_uv, tf_dict)
        loss_f_s = self.sess.run(self.loss_f_s, tf_dict)
        loss_HOLE = self.sess.run(self.loss_HOLE, tf_dict)
        loss = self.sess.run(self.loss, tf_dict)
        loss_PART = self.sess.run(self.loss_PART, tf_dict)
        loss_DIST = self.sess.run(self.loss_DIST, tf_dict)
        print('loss_f_uv', loss_f_uv)
        print('loss_f_s', loss_f_s)
        print('loss_HOLE', loss_HOLE)
        print('loss', loss)
        print('loss_PART', loss_PART)
        print('loss_DIST', loss_DIST)