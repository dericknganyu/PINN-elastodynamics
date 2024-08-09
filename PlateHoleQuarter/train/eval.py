import numpy as np
import time
from pyDOE import lhs

import shutil
import math

# from silence_tensorflow import silence_tensorflow
# silence_tensorflow()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
# Setup GPU for training

# import tensorflow.compat.v1 as tf # type: ignore
# tf.disable_v2_behavior()
import tensorflow as tf
# import deepxde as dde
# import tensorflow as tff
# import tensorflow_estimator as estimator

import wandb
wandb.require("core")
# wandb.init()
# wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)

from models import *
from utils import *

import time


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(1111)
tf.set_random_seed(1111)

tf.get_logger().setLevel(3)
tf.autograph.set_verbosity(3)
tf.logging.set_verbosity(tf.logging.ERROR)
# Note: TensorFlow 1.10 version is used

time_var = '20240807-202953'
run_suffix = 154000#"Z_end"
run_prefix = ''

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

    direct = '../output/'+time_var
    if not os.path.exists(direct):
        os.makedirs(direct)
    pts_plot([XYT_c, IC, LW, UP, LF, RT, HOLE], direct)
    
    with tf.device('/device:GPU:0'):      

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
         
        
        direct = '../PINN_result/'+time_var
        if not os.path.exists(direct):
            os.makedirs(direct)

        # Provide directory (second init) for pretrained networks if you have
        model = PINN(XYT_c, HOLE, IC, LF, RT, UP, LW, DIST, uv_layers, dist_layers, part_layers, lb, ub, direct,
                        partDir='/partNN_float64.pickle', distDir='/distNN_float64.pickle', uvDir='/%suvNN_float64.pickle_then_%s'%(run_prefix, run_suffix))


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

        ############################################################
        ######### Plot the stress distr on the notch ###############
        ############################################################
        direct = '../results/'+time_var
        if not os.path.exists(direct):
            os.makedirs(direct)
        theta = np.linspace(0.0, np.pi / 2.0, 100)
        TIME  = [2.5   , 3.75   , 5.0]
        COLOR = ['blue', 'green', 'red']
        quantity_dict = {'s11':'sigma_{11}', 's22':'sigma_{22}', 's12':'sigma_{12}'}
        for quantity, name_quan in quantity_dict.items():
        # for quantity_dict in QUANT:
            # do_plot(TIME, theta, model, quantity, name_quan)
            do_plot(TIME, theta, model, quantity, name_quan, COLOR, path=direct)



        direct = '../output/'+time_var
        if not os.path.exists(direct):
            os.makedirs(direct)
            
        # N_x = x_star.size
        # t_star = np.linspace(0, MAX_T, N_t).reshape(-1, 1)
        # t_star = np.repeat(t_star, N_x, axis=0)
        # x_star = np.tile(x_star, (N_t, 1))
        # y_star = np.tile(y_star, (N_t, 1))
        
        # t0 = time.time()
        # u_pred, v_pred, s11_pred, s22_pred, s12_pred, e11_pred, e22_pred, e12_pred = model.predict(x_star, y_star, t_star)
        # t1 = time.time()
        # print(t1-t0)
        # for i in range(N_t):
        #     start = i*N_x
        #     start  = (i+1)*N_x
        #     field = [x_star[start:start, :], y_star[start:start, :], t_star[start:start, :], u_pred[start:start, :], v_pred[start:start, :], s11_pred[start:start, :], s22_pred[start:start, :], s12_pred[start:start, :]]
        #     amp_pred = (u_pred ** 2 + v_pred ** 2) ** 0.5
        #     print('Inferring for frame %s of %s'%(i, N_t))
        #     postProcessDef(xmin=0, xmax=0.50, ymin=0, ymax=0.50, num=i, s=4, scale=0, field=field, path=direct)
        offset = 0.04
        for i in range(N_t):
            t_star = np.zeros((x_star.size, 1))
            t_star.fill(i * MAX_T / (N_t - 1))
            t0 = time.time()
            u_pred, v_pred, s11_pred, s22_pred, s12_pred, e11_pred, e22_pred, e12_pred = model.predict(x_star, y_star, t_star)
            t1 = time.time()
            print(t1-t0)
            field = [x_star, y_star, t_star, u_pred, v_pred, s11_pred, s22_pred, s12_pred]
            amp_pred = (u_pred ** 2 + v_pred ** 2) ** 0.5
            print('Inferring for frame %s of %s'%(i, N_t))
            postProcessDef(xmin=0-offset, xmax=0.50+offset, 
                           ymin=0-offset, ymax=0.50+offset, 
                           num=i, s=4, scale=1, field=field, path=direct)    
        
        name = 'stress_comparison_{}.png'
        create_gif(name, direct, 'a_'+name[0:-7]+'.gif')

        name = 'uv_comparison_{}.png'
        create_gif(name, direct, 'a_'+name[0:-7]+'.gif')