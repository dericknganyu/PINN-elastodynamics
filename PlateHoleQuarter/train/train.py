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
    XYT_c = lb + (ub - lb) * lhs(3, 7000)
    XYT_c_ref = lb + np.array([0.15, 0.15, 10.0]) * lhs(3, 4000)  # Refinement for stress concentration
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
    
    with tf.device('/device:GPU:%s'%(DEVICE)):      

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        # wandb.tensorflow.log(tf.summary.merge_all())  
        
        direct = '../PINN_result/'+time_var
        if not os.path.exists(direct):
            os.makedirs(direct)

        # Provide directory (second init) for pretrained networks if you have
        model = PINN(XYT_c, HOLE, IC, LF, RT, UP, LW, DIST, uv_layers, dist_layers, part_layers, lb, ub, direct)
        # model = PINN(XYT_c, HOLE, IC, LF, RT, UP, LW, DIST, uv_layers, dist_layers, part_layers, lb, ub,
        #                 partDir='./partNN_float64.pickle', distDir='./distNN_float64.pickle', uvDir='uvNN_float64.pickle')
                
        
        # Train the distance function
        print('Training distance function')
        wandb.init()
        model.train_bfgs_dist()
        model.save_NN('%s/distNN_float64.pickle'%(direct), TYPE='DIST')
        model.count = 0
        wandb.finish()
        # wandb.run.save()
        # wandb_name = wandb.run.name
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