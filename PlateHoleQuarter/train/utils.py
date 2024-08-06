import numpy as np
import matplotlib
import platform
if platform.system()=='Linux':
    matplotlib.use('Agg')
if platform.system()=='Windows':
    from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import scipy.io

# Setup GPU for training
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


import tensorflow.compat.v1 as tf # type: ignore
tf.disable_v2_behavior()
import tensorflow as tf1

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
np.random.seed(1111)
tf.set_random_seed(1111)



# tf.logging.set_verbosity(tf.logging.ERROR)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Note: TensorFlow 1.10 version is used


def GenDistPt(xmin, xmax, ymin, ymax, tmin, tmax, xc, yc, r, num_surf_pt, num, num_t):
    # num: number per edge
    # num_t: number time step
    x = np.linspace(xmin, xmax, num=num)
    y = np.linspace(ymin, ymax, num=num)
    x, y = np.meshgrid(x, y)
    # Delete point in hole
    dst = ((x - xc) ** 2 + (y - yc) ** 2) ** 0.5
    x = x[dst >= r]
    y = y[dst >= r]
    x = x.flatten()[:, None]
    y = y.flatten()[:, None]
    # Refinement point near hole surface
    theta = np.linspace(0.0, np.pi / 2.0, num_surf_pt)
    x_surf = np.multiply(r, np.cos(theta)) + xc
    y_surf = np.multiply(r, np.sin(theta)) + yc
    x_surf = x_surf.flatten()[:, None]
    y_surf = y_surf.flatten()[:, None]
    x = np.concatenate((x, x_surf), 0)
    y = np.concatenate((y, y_surf), 0)
    # Cartisian product with time points
    t = np.linspace(tmin, tmax, num=num_t)
    xxx, ttt = np.meshgrid(x, t)
    yyy,   _ = np.meshgrid(y, t)
    xxx = xxx.flatten()[:, None]
    yyy = yyy.flatten()[:, None]
    ttt = ttt.flatten()[:, None]
    return xxx, yyy, ttt

def GenDist(XYT_dist):
    dist_u = np.zeros_like(XYT_dist[:, 0:1])
    dist_v = np.zeros_like(XYT_dist[:, 0:1])
    dist_s11 = np.zeros_like(XYT_dist[:, 0:1])
    dist_s22 = np.zeros_like(XYT_dist[:, 0:1])
    dist_s12 = np.zeros_like(XYT_dist[:, 0:1])
    for i in range(len(XYT_dist)):
        dist_u[i, 0] = min(XYT_dist[i][2], XYT_dist[i][0])  # min(t, x-(-0.5))
        dist_v[i, 0] = min(XYT_dist[i][2], XYT_dist[i][1])  # min(t, sqrt((x+0.5)^2+(y+0.5)^2))
        dist_s11[i, 0] = min(XYT_dist[i][2], 0.5 - XYT_dist[i][0])
        dist_s22[i, 0] = min(XYT_dist[i][2], 0.5 - XYT_dist[i][1])  # min(t, 0.5-y, 0.5+y)
        dist_s12[i, 0] = min(XYT_dist[i][2], XYT_dist[i][1], 0.5 - XYT_dist[i][1], XYT_dist[i][0], 0.5 - XYT_dist[i][0])
    DIST = np.concatenate((XYT_dist, dist_u, dist_v, dist_s11, dist_s22, dist_s12), 1)
    return DIST

def preprocess(dir):
    # dir: directory of ground truth
    data = scipy.io.loadmat(dir)
    X = data['x']
    Y = data['y']
    Exact_u = data['u']
    Exact_v = data['v']
    Exact_s11 = data['s11']
    Exact_s22 = data['s22']
    Exact_s12 = data['s12']
    # Flatten to be Nx1
    x_star = X.flatten()[:, None]
    y_star = Y.flatten()[:, None]
    u_star = Exact_u.flatten()[:, None]
    v_star = Exact_v.flatten()[:, None]
    s11_star = Exact_s11.flatten()[:, None]
    s22_star = Exact_s22.flatten()[:, None]
    s12_star = Exact_s12.flatten()[:, None]
    return x_star, y_star, u_star, v_star, s11_star, s22_star, s12_star

def postProcessDef(xmin, xmax, ymin, ymax, field, path, s=5, num=0, scale=1):
    ''' Plot deformed plate (set scale=0 want to plot undeformed contours)
    '''
    [x_star, y_star, u_star, v_star, s11_star, s22_star, s12_star] = preprocess(
        '../FEM_result/Quarter_plate_hole_dynamic/ProbeData-' + str(num) + '.mat')       # FE solution
    [x_pred, y_pred, _, u_pred, v_pred, s11_pred, s22_pred, s12_pred] = field
    #
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.2, wspace=0.3)
    x_PINN = x_pred + u_pred*scale
    y_PINN = y_pred + v_pred*scale
    x_FEM  = x_star + u_star*scale
    y_FEM  = y_star + v_star*scale

    X_list = [x_PINN    , x_PINN    , x_FEM    , x_FEM    ]
    Y_list = [y_PINN    , y_PINN    , y_FEM    , y_FEM    ]
    C_list = [u_pred    , v_pred    , u_star   , v_star   ]
    LABELS = ['$u$-PINN', '$v$-PINN', '$u$-FEM', '$v$-FEM']
    M_SIZE = [s         , s         , 2+s         , 2+s   ]

    for axs in ax.flat:
        i = 0
        cf = axs.scatter(X_list[i], Y_list[i], c=C_list[i], alpha=0.7, edgecolors='none',
                            cmap='rainbow', marker='o', s=M_SIZE[i], vmin=0, vmax=0.04)
        # cf.cmap.set_under('whitesmoke')
        # cf.cmap.set_over('black')
        axs.set_title(r''+LABELS[i], fontsize=16)
        cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=axs)
        cbar.ax.tick_params(labelsize=14)

        axs.axis('square')
        for key, spine in axs.spines.items():
            if key in ['right', 'top', 'left', 'bottom']:
                spine.set_visible(False)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_xlim([xmin, xmax])
        axs.set_ylim([ymin, ymax])

        i = i+1

    plt.savefig('%s/uv_comparison_'%(path) + str(num) + '.png', dpi=200)
    plt.close('all')
    #
    # Plot predicted stress
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))
    fig.subplots_adjust(hspace=0.15, wspace=0.3)
    X_list = [x_PINN               , x_PINN               , x_PINN               , x_FEM                , x_FEM                , x_FEM              ]
    Y_list = [y_PINN               , y_PINN               , y_PINN               , y_FEM                , y_FEM                , y_FEM              ]
    C_list = [s11_pred             , s22_pred             , s12_pred             , s11_star             , s22_star             , s12_star           ]
    LABELS = ['$\sigma_{11}$-PINN' ,'$\sigma_{22}$-PINN'  ,'$\sigma_{12}$-PINN'  , '$\sigma_{11}$-FEM'  ,'$\sigma_{22}$-FEM'   ,'$\sigma_{12}$-FEM' ]
    M_SIZE = [s                    , s                    , s                    , 2+s                  , 2+s                  , 2+s                ]
    for axs in ax.flat:
        i = 0
        cf = axs.scatter(X_list[i], Y_list[i], c=C_list[i], alpha=0.7, edgecolors='none',
                            cmap='rainbow', marker='o', s=M_SIZE[i], vmin=0, vmax=0.04)
        # cf.cmap.set_under('whitesmoke')
        # cf.cmap.set_over('black')
        axs.set_title(r'$\sigma_{11}$-PINN', fontsize=16)
        cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=axs)
        cbar.ax.tick_params(labelsize=14)
        
        axs.axis('square')
        for key, spine in axs.spines.items():
            if key in ['right', 'top', 'left', 'bottom']:
                spine.set_visible(False)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_xlim([xmin, xmax])
        axs.set_ylim([ymin, ymax])
        
    plt.savefig('%s/stress_comparison_'%(path) + str(num) + '.png', dpi=200)
    plt.close('all')

def DelHolePT(XYT_c, xc=0, yc=0, r=0.1):
    # Delete points within hole
    dst = np.array([((xyt[0] - xc) ** 2 + (xyt[1] - yc) ** 2) ** 0.5 for xyt in XYT_c])
    return XYT_c[dst > r, :]

def GenHoleSurfPT(xc, yc, r, N_PT):
    # Generate
    theta = np.linspace(0.0, np.pi / 2.0, N_PT)
    xx = np.multiply(r, np.cos(theta)) + xc
    yy = np.multiply(r, np.sin(theta)) + yc
    xx = xx.flatten()[:, None]
    yy = yy.flatten()[:, None]
    return xx, yy

def getPINNsol(time, theta, model):
    r = 0.1
    x_surf = np.multiply(r, np.cos(theta))
    y_surf = np.multiply(r, np.sin(theta))
    x_surf = x_surf.flatten()[:, None]
    y_surf = y_surf.flatten()[:, None]
    t_surf = 0 * y_surf + time
    u_surf, v_surf, s11_surf, s22_surf, s12_surf, _, _, _ = model.predict(x_surf, y_surf, t_surf)
    return u_surf, v_surf, s11_surf, s22_surf, s12_surf
#
def getFEMsol(time):
    num = int(time / 0.125)  # num of frame
    [x_star, y_star, u_star, v_star, s11_star, s22_star, s12_star] = preprocess(
        '../FEM_result/Quarter_plate_hole_dynamic/ProbeData-' + str(num) + '.mat')
    mask = (x_star ** 2 + y_star ** 2) <= 0.010001
    return x_star[mask], y_star[mask], s11_star[mask], s22_star[mask], s12_star[mask]


def compare_plot(time, theta, model, quantity):
    _,      _, s11_surf, s22_surf, s12_surf = getPINNsol(time, theta, model)
    x_star, _, s11_star, s22_star, s12_star = getFEMsol(time)

    if quantity == 's11':
        plt.plot(theta * 180 / np.pi, s11_surf, '-', alpha=0.8, label='t=%ss, PINN'%(time))
        plt.scatter(np.arccos(x_star / 0.1) * 180 / np.pi, s11_star, marker='^', s=7, alpha=1, label='t=%ss, FEM'%(time))
    if quantity == 's22':
        plt.plot(theta * 180 / np.pi, s22_surf, '-', alpha=0.8, label='t=%ss, PINN'%(time))
        plt.scatter(np.arccos(x_star / 0.1) * 180 / np.pi, s22_star, marker='^', s=7, alpha=1, label='t=%ss, FEM'%(time))
    if quantity == 's12':
        plt.plot(theta * 180 / np.pi, s12_surf, '-', alpha=0.8, label='t=%ss, PINN'%(time))
        plt.scatter(np.arccos(x_star / 0.1) * 180 / np.pi, s12_star, marker='^', s=7, alpha=1, label='t=%ss, FEM'%(time))

# def do_plot(TIME, theta, model, quantity_dict):
def do_plot(TIME, theta, model, quantity, name_quan, path):
    #
    # Plot stress distribution for s11 on circular surf
    # quantity, name_quan  = quantity_dict.items()#list(quantity_dict.items())[0]
    plt.figure(figsize=(5, 5))
    for time in TIME:
        compare_plot(time, theta, model, quantity)

    plt.xlim([0, 90])
    plt.xticks([0, 30, 60, 90], fontsize=11)
    plt.ylim([-0.5, 3.5])
    plt.yticks([0, 1, 2, 3], fontsize=11)
    plt.xlabel(r'$\theta$/degree', fontsize=12)
    plt.ylabel(r'$\%s$\kPa'%(name_quan), fontsize=12)
    plt.legend(fontsize=12, frameon=False)
    plt.savefig('%s/%s_comparison.png'%(path, quantity),dpi=300)
    # plt.show()

def pts_plot(points, path):

    # Visualize ALL the training points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for point in points:
        ax.scatter(point[:,0:1], point[:,1:2], point[:,2:3], marker='o', alpha=0.1, s=2)#, color='blue')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('T axis')
    plt.savefig('%s/points.png'%(path),dpi=300)

    plt.show()

import plotly.graph_objects as go
import numpy as np

def pts_plot_interactive(points, path):
    # Create the 3D scatter plot
    fig = go.Figure()

    for point in points:
        fig.add_trace(go.Scatter3d(
            x=point[:,0:1], 
            y=point[:,1:2], 
            z=point[:,2:3], 
            mode='markers',
            marker=dict(size=2, opacity=0.1)
        ))

    fig.update_layout(
        scene = dict(
            xaxis_title='X axis',
            yaxis_title='Y axis',
            zaxis_title='T axis'
        )
    )

    # Save the interactive plot as an HTML file
    fig.write_html(f'{path}/points.html')

    # Show the plot
    fig.show()

from natsort import natsorted

def list_filtered_files_in_directory(directory_path, filter_string):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return []

    # List all files and directories in the specified path
    files_and_dirs = os.listdir(directory_path)
    
    # Filter out directories, keep only files that contain the filter string
    filtered_files = [
        f for f in files_and_dirs 
        if os.path.isfile(os.path.join(directory_path, f)) and filter_string in f
    ]
    
    # Sort the filtered files
    filtered_files = natsorted(filtered_files)
    
    return filtered_files

def delete_files_if_exceeding_threshold(directory_path, filter_string, threshold):
    # Get the list of filtered and sorted files
    filtered_files = list_filtered_files_in_directory(directory_path, filter_string)
    
    # Count the number of filtered files
    n = len(filtered_files)
    
    # If the number of files exceeds the threshold m, delete the first n-m files
    if n > threshold:
        files_to_delete = filtered_files[:n-threshold]
        for file in files_to_delete:
            file_path = os.path.join(directory_path, file)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
                
    return filtered_files[n-threshold:] if n > threshold else filtered_files