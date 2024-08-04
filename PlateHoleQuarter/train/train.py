import numpy as np
import time
from pyDOE import lhs

import shutil
import math

# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# Setup GPU for training
import tensorflow.compat.v1 as tf # type: ignore
tf.disable_v2_behavior()
import deepxde as dde
import tensorflow as tff

import wandb
# wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)

from models import *
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(1111)
tf.set_random_seed(1111)

# Note: TensorFlow 1.10 version is used


if __name__ == "__main__":

    #### Note: The detailed description for this case can be found in paper:
    #### Physics informed deep learning for computational elastodynamicswithout labeled data.
    #### https://arxiv.org/abs/2006.08472
    #### But network configuration might be slightly different from what is described in paper.
    PI = math.pi
    MAX_T = 10.0

    # Domain bounds for x, y and t
    lb = np.array([0, 0, 0.0])
    ub = np.array([0.5, 0.5, 10.0])

    # Network configuration
    uv_layers   = [3] + 8 * [70] + [5]
    dist_layers = [3] + 4 * [20] + [5]
    part_layers = [3] + 4 * [20] + [5]

    # Number of frames for postprocessing
    N_t = int(MAX_T * 8 + 1)

    # Generate distance function for spatio-temporal space
    x_dist, y_dist, t_dist = GenDistPt(xmin=0, xmax=0.5, ymin=0, ymax=0.5, tmin=0, tmax=10, xc=0, yc=0, r=0.1,
                                       num_surf_pt=40, num=21, num_t=21)
    XYT_dist = np.concatenate((x_dist, y_dist, t_dist), 1)
    DIST = GenDist(XYT_dist)

    # Initial condition point for u, v
    IC = lb + np.array([0.5, 0.5, 0.0]) * lhs(3, 5000)
    IC = DelHolePT(IC, xc=0, yc=0, r=0.1)

    # Collocation point for equation residual
    XYT_c = lb + (ub - lb) * lhs(3, 70000)
    XYT_c_ref = lb + np.array([0.15, 0.15, 10.0]) * lhs(3, 40000)  # Refinement for stress concentration
    XYT_c = np.concatenate((XYT_c, XYT_c_ref), 0)
    XYT_c = DelHolePT(XYT_c, xc=0, yc=0, r=0.1)

    xx, yy = GenHoleSurfPT(xc=0, yc=0, r=0.1, N_PT=83)
    tt = np.linspace(0, 10, 121)
    tt = tt[1:]
    x_ho, t_ho = np.meshgrid(xx, tt)
    y_ho, _ = np.meshgrid(yy, tt)
    x_ho = x_ho.flatten()[:, None]
    y_ho = y_ho.flatten()[:, None]
    t_ho = t_ho.flatten()[:, None]
    HOLE = np.concatenate((x_ho, y_ho, t_ho), 1)

    LW = np.array([0.1, 0.0, 0.0]) + np.array([0.4, 0.0, 10]) * lhs(3, 8000)
    UP = np.array([0.0, 0.5, 0.0]) + np.array([0.5, 0.0, 10]) * lhs(3, 8000)
    LF = np.array([0.0, 0.1, 0.0]) + np.array([0.0, 0.4, 10]) * lhs(3, 8000)
    RT = np.array([0.5, 0.0, 0.0]) + np.array([0.0, 0.5, 10]) * lhs(3, 13000)

    t_RT = RT[:, 2:3]
    period = 5  # two period in 10s
    s11_RT = 0.5 * np.sin((2 * PI / period) * t_RT + 3 * PI / 2) + 0.5
    RT = np.concatenate((RT, s11_RT), 1)

    # Add some boundary points into the collocation point set
    XYT_c = np.concatenate((XYT_c, HOLE[::4, :], LF[::5, :], RT[::5, 0:3], UP[::5, :], LW[::5, :]), 0)

    direct = '../output'
    if not os.path.exists(direct):
        os.makedirs(direct)
    pts_plot([XYT_c, IC, LW, UP, LF, RT, HOLE], direct)
    
    with tf.device('/device:GPU:0'):      

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        # wandb.tensorflow.log(tf.summary.merge_all())  

        # Provide directory (second init) for pretrained networks if you have
        model = PINN(XYT_c, HOLE, IC, LF, RT, UP, LW, DIST, uv_layers, dist_layers, part_layers, lb, ub)
        # model = PINN(XYT_c, HOLE, IC, LF, RT, UP, LW, DIST, uv_layers, dist_layers, part_layers, lb, ub,
        #                 partDir='./partNN_float64.pickle', distDir='./distNN_float64.pickle', uvDir='uvNN_float64.pickle')

        # Train the distance function
        print('Training distance function')
        model.train_bfgs_dist()
        model.count = 0
        print()

        # Train the NN for particular solution
        print('Training particular function')
        model.train_bfgs_part()
        model.count = 0
        print()

        # Train the composite network
        start_time = time.time()
        # model.train(iter=1000, learning_rate=5e-4)
        print('Training General function')
        model.train_bfgs()
        print("--- %s seconds ---" % (time.time() - start_time))
        print()

        # Save the trained model
        direct = '../PINN_result'
        if not os.path.exists(direct):
            os.makedirs(direct)
        model.save_NN('%s/uvNN_float64.pickle'%(direct), TYPE='UV')
        model.save_NN('%s/distNN_float64.pickle'%(direct), TYPE='DIST')
        model.save_NN('%s/partNN_float64.pickle'%(direct), TYPE='PART')

        # Check the loss for each part
        model.getloss()

        # Output result at each time step
        x_star = np.linspace(0, 0.5, 251)
        y_star = np.linspace(0, 0.5, 251)
        x_star, y_star = np.meshgrid(x_star, y_star)
        x_star = x_star.flatten()[:, None]
        y_star = y_star.flatten()[:, None]
        dst = ((x_star - 0) ** 2 + (y_star - 0) ** 2) ** 0.5
        x_star = x_star[dst >= 0.1]
        y_star = y_star[dst >= 0.1]
        x_star = x_star.flatten()[:, None]
        y_star = y_star.flatten()[:, None]
        # shutil.rmtree('../output', ignore_errors=True)
        # os.makedirs('../output')
        direct = '../output'
        if not os.path.exists(direct):
            os.makedirs(direct)
        for i in range(N_t):
            t_star = np.zeros((x_star.size, 1))
            t_star.fill(i * MAX_T / (N_t - 1))
            u_pred, v_pred, s11_pred, s22_pred, s12_pred, e11_pred, e22_pred, e12_pred = model.predict(x_star, y_star, t_star)
            field = [x_star, y_star, t_star, u_pred, v_pred, s11_pred, s22_pred, s12_pred]
            amp_pred = (u_pred ** 2 + v_pred ** 2) ** 0.5
            postProcessDef(xmin=0, xmax=0.50, ymin=0, ymax=0.50, num=i, s=4, scale=0, field=field, path=direct)


        ############################################################
        ######### Plot the stress distr on the notch ###############
        ############################################################
        direct = '../results'
        if not os.path.exists(direct):
            os.makedirs(direct)
        theta = np.linspace(0.0, np.pi / 2.0, 100)
        TIME  = [2.5, 3.75, 5.0]
        quantity_dict = {'s11':'sigma_{11}', 's22':'sigma_{22}', 's12':'sigma_{12}'}
        for quantity, name_quan in quantity_dict.items():
        # for quantity_dict in QUANT:
            # do_plot(TIME, theta, model, quantity, name_quan)
            do_plot(TIME, theta, model, quantity, name_quan, path=direct)
