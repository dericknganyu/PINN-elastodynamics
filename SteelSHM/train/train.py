DEVICE = '1'

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
import tensorflow as tf1
# import deepxde as dde
# import tensorflow_estimator as estimator

import wandb # type: ignore
wandb.require("core")
wandb.init()
# wandb.init(config=tf.flags.FLAGS, sync_tensorboard=True)

from models import *
from utils import *
from utils_shm import *

import time
time_var = time.strftime('%Y%m%d-%H%M%S')
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE
np.random.seed(1111)
tf.set_random_seed(1111)


# tf.get_logger().setLevel('NONE')
# tf.autograph.set_verbosity(3)
# tf.logging.set_verbosity(tf.logging.ERROR)
# tf.logging.set_verbosity(tf.logging.ERROR)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Note: TensorFlow 1.10 version is used


if __name__ == "__main__":

    #### Note: The detailed description for this case can be found in paper:
    #### Physics informed deep learning for computational elastodynamicswithout labeled data.
    #### https://arxiv.org/abs/2006.08472
    #### But network configuration might be slightly different from what is described in paper.
    PI = math.pi
    nondim = True

    w = 0.3
    h = 0.004
    in_size = 9529
    in_size = 9529
    lr_size = 512 #1024#128
    tb_size = 512 #int(lr_size/2)

    rho = 7.9e3 #7900
    E = 1.8e11 #180e9
    nu = 0.3
    mu0 = E/(2*(1+nu))
    lambda0 = E*nu/((1+nu)*(1-2*nu))

    V0 = 1e-12 #1e-8
    f0 = 120e3
    n0 = 5
    t_tot = 4.167e-5
    tmax = 2.083e-4  

    T = tmax  

    # Domain bounds for x, y and t
    lb = np.array([0, 0, 0.0])
    ub = np.array([w, h, tmax])

    # Network configuration
    uv_layers   = [3] + 8 * [70] + [5]
    dist_layers = [3] + 4 * [20] + [5]
    part_layers = [3] + 4 * [20] + [5]

    # Number of frames for postprocessing

    # Initial condition point for u, v
    IC = lb + np.array([w, h, 0.0]) * lhs(3, 5000)

    # Collocation point for equation residual
    XYT_c = lb + (ub - lb) * lhs(3, 7000)
    
    # P1 = np.array([0, 0, 0.0])
    # P2 = np.array([0, h, 0.0])
    # P3 = np.array([w, h/2, 0.0])
    dense = 10
    XYT_P1 = np.array([0        , 0          , 0]) + np.array([w/dense, h/dense  , T]) * lhs(3, 4000) 
    XYT_P2 = np.array([0        , h  -h/dense, 0]) + np.array([w/dense, h/dense  , T]) * lhs(3, 4000) 
    XYT_P3 = np.array([w-w/dense, h/2-h/dense, 0]) + np.array([w/dense, 2*h/dense, T]) * lhs(3, 8000) 

    XYT_c = np.concatenate((XYT_c, XYT_P1, XYT_P2, XYT_P3), 0)

    LW = np.array([0, 0, 0]) + np.array([w, 0, T]) * lhs(3, 8000)
    UP = np.array([0, h, 0]) + np.array([w, 0, T]) * lhs(3, 8000)
    LF = np.array([0, 0, 0]) + np.array([0, h, T]) * lhs(3, 8000)
    RT = np.array([w, 0, 0]) + np.array([0, h, T]) * lhs(3, 8000)

    P1 = np.array([0, 0  , 0]) + np.array([0, 0, T]) * lhs(3, 4000)
    P2 = np.array([0, h  , 0]) + np.array([0, 0, T]) * lhs(3, 4000)
    P3 = np.array([w, h/2, 0]) + np.array([0, 0, T]) * lhs(3, 4000)

    t_P1 = P1[:, 2:3]
    t_P2 = P2[:, 2:3]

    v_P1 = f_exc(P1[:,1:2], P1[:,2:3])
    v_P2 = f_exc(P2[:,1:2], P2[:,2:3])
    # v_P3 = np.zeros_like(P1[:,1:2])

    # u_P1 = np.zeros_like(P1[:,1:2])
    # u_P2 = np.zeros_like(P2[:,1:2])
    # u_P3 = np.zeros_like(P1[:,1:2])

    P1 = np.concatenate((P1, v_P1), 1)
    P2 = np.concatenate((P2, v_P2), 1)

    # Add some boundary points into the collocation point set
    XYT_c = np.concatenate((XYT_c, LF[::5, :], RT[::5, :], UP[::5, :], LW[::5, :], P1[::5, 0:3], P2[::5, 0:3], P3[::5, :]), 0)

    direct = '../output/'+time_var
    if not os.path.exists(direct):
        os.makedirs(direct)
    pts_plot([XYT_c, IC, LW, UP, LF, RT, P1, P2, P3], direct, 'before_nondim')
    x_dist, y_dist, t_dist = GenDistPt(xmin=0, xmax=w, ymin=0, ymax=h, tmin=0, tmax=T,
                                       num=21, num_t=21)
    XYT_dist = np.concatenate((x_dist, y_dist, t_dist), 1)
    DIST, _ = GenDist(XYT_dist, w, h)
    plot_distance(w, h, tmax, direct, 'before_nondim')

    if nondim == True:
        # Nondimensionalization parameters
        L_star = 0.3 
        T_star = L_star*np.sqrt(rho/mu0) 
        U_star = V0
        S_star = rho*L_star*U_star/(T_star**2)

        var_star = np.array([L_star, L_star, T_star])
        var_star_U = np.array([L_star, L_star, T_star, U_star])
        # Nondimensionalize coordinates and inflow velocity
        XYT_c /= var_star 
        P1 /= var_star_U
        P2 /= var_star_U
        P3 /= var_star 
        IC /= var_star 
        LF /= var_star 
        RT /= var_star 
        UP /= var_star 
        LW /= var_star 
        lb /= var_star 
        ub /= var_star 

        w /= L_star
        h /= L_star
        T /= T_star
        tmax /= T_star
        
    else:
        L_star = 1.0
        T_star = 1.0
        U_star = 1.0
        S_star = 1.0

    
    # Generate distance function for spatio-temporal space
    pts_plot([XYT_c, IC, LW, UP, LF, RT, P1, P2, P3], direct, 'after_nondim')
    x_dist, y_dist, t_dist = GenDistPt(xmin=0, xmax=w, ymin=0, ymax=h, tmin=0, tmax=T,
                                       num=21, num_t=21)
    XYT_dist = np.concatenate((x_dist, y_dist, t_dist), 1)
    DIST, _ = GenDist(XYT_dist, w, h)
    plot_distance(w, h, tmax, direct, 'after_nondim')
    
    with tf.device('/device:GPU:%s'%(DEVICE)):      

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        # wandb.tensorflow.log(tf.summary.merge_all())  
        
        direct = '../PINN_result/'+time_var
        if not os.path.exists(direct):
            os.makedirs(direct)

        # Provide directory (second init) for pretrained networks if you have
        model = PINN(XYT_c, P1, P2, P3, IC, LF, RT, UP, LW, DIST, 
                     uv_layers, dist_layers, part_layers, 
                     lb, ub, 
                     rho, mu0, lambda0, 
                     L_star, T_star, U_star, S_star, 
                     direct)
        # model = PINN(XYT_c, HOLE, IC, LF, RT, UP, LW, DIST, uv_layers, dist_layers, part_layers, lb, ub,
        #                 partDir='./partNN_float64.pickle', distDir='./distNN_float64.pickle', uvDir='uvNN_float64.pickle')
                
        
        # Train the distance function
        print('Training distance function')
        wandb.init()
        model.train_bfgs_dist()
        model.save_NN('%s/distNN_float64.pickle'%(direct), TYPE='DIST')
        model.count = 0
        wandb.finish()
        print()

        # Train the NN for particular solution
        print('Training particular function')
        wandb.init()#name = wandb_name+'_Part')
        model.train_bfgs_part()
        model.save_NN('%s/partNN_float64.pickle'%(direct), TYPE='PART')
        model.count = 0
        wandb.finish()
        print()

        # Train the composite network
        start_time = time.time()
        wandb.init()#name = wandb_name+'_Gen')
        print('Training General function')
        model.train(iter=2000, learning_rate=1e-3)
        model.save_NN('%s/zAdam1_uvNN_float64.pickle'%(direct), TYPE='UV')
        print('Training General function')
        model.train(iter=2000, learning_rate=5e-4)
        model.save_NN('%s/zAdam2_uvNN_float64.pickle'%(direct), TYPE='UV')
        print('Training General function')
        model.train_bfgs()
        model.save_NN('%s/uvNN_float64.pickle_Z_end'%(direct), TYPE='UV')
        print("--- %s seconds ---" % (time.time() - start_time))
        print()
        # wandb.finish()

        # Save the trained model


        # Check the loss for each part
        model.getloss()