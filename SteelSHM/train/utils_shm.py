import numpy as np
import utils
import matplotlib.pyplot as plt


dir_path = '/home/derick/Documents/PINNS/examples/mfl_src/'

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
# mu0 = 1.8e11 #180e9
# lambda0 = 0.3

V0 = 1e-12 #1e-8
f0 = 120e3
n0 = 5
t_tot = 4.167e-5
tmax = 2.083e-4

density = 10


def hanning_window(t, T):
    # return (torch.sin(math.pi * t / T))**2  
    return 0.5 * (1 - np.cos(2 * np.pi * t / T))

def hanning_windowed_sine_burst(t, A, fc, T):
    # A       # Amplitude of the sine wave
    # fc      # Frequency of the sine wave (in Hz)
    # T       # Duration of the burst (in seconds)
    return A * np.sin(2 * np.pi * fc * t) * hanning_window(t, T)

def f_exc(z, t, sym=False, V=V0, f=f0, n=n0, t_tot=t_tot):
    cut_off_val = hanning_windowed_sine_burst(np.array(t_tot), V, f, t_tot)
    step_fn = np.heaviside(t_tot-t, cut_off_val)
    # step_fn = 1

    if sym:
        # Ensure that excitation at f_exc(z=0,t) = -f_exc(z=h,t)
        result = np.cos(np.pi*z/h)*hanning_windowed_sine_burst(t, V, f, t_tot)*step_fn
    else:
        result = hanning_windowed_sine_burst(t, V, f, t_tot)*step_fn

    return result

def plot_distance(w, h, tmax, path, track):
    x = np.linspace(0, w, 100)
    z = np.linspace(0, h, 100)
    x, z = np.meshgrid(x, z)

    x = x.flatten()[:, None]
    z = z.flatten()[:, None]

    tval = tmax * 0.5 * np.ones_like(x)
    XYT_dist = np.concatenate((x, z, tval), 1)

    _, values  = utils.GenDist(XYT_dist, w, h)
    Du_values, Dv_values, Dxx_values, Dzz_values, Dxz_values = values
    Du_values  = Du_values.reshape(100, 100)
    Dv_values  = Dv_values.reshape(100, 100)
    Dxx_values = Dxx_values.reshape(100, 100)
    Dzz_values = Dzz_values.reshape(100, 100)
    Dxz_values = Dxz_values.reshape(100, 100)


    cmap = 'rainbow'#'jet'#'viridis'#'turbo'

    D_vals = {
        'Dxx': Dxx_values,
        'Dzz': Dzz_values,
        'Dxz': Dxz_values,
        'Du': Du_values,
        'Dv': Dv_values,
        'Dxv': Dv_values,
    }

    fig, axs = plt.subplots(2, 3, figsize=(18, 6), constrained_layout=True)

    i = 0
    for ax, (key, D_values) in zip(axs.flat, D_vals.items()):
        ax.set_axis_off()
        if i == 5:
            continue
        im = ax.imshow(D_values, extent=[0, w, 0, h], origin='lower', cmap=cmap, aspect='auto')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(f'Image plot of {key}')

        # Create a colorbar with the same height as the imshow
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        i += 1

    plt.savefig('%s/distance_fxn%s.png'%(path, track),dpi=300)
    plt.show()   


# class distance_fxn():
#     def __init__(self, boundary):#, domain, details):
#         # seed_everything(0, workers=True)

#         self.batch_line_dist  = vmap(self.line_dist,(0, 0, 0))
#         self.batch_plane_dist = vmap(self.plane_dist,(0, 0, 0))

#         if boundary == 'u':
#             self.max_val = tmax#w/2
#             # x = 0, z = 0 & x = 0, z = h
#             # x = w, z = h/2
#             # t = 0
#             # x = 0
#             self.xtics = {'lines' :{'A':np.array([[0, 0, 0], [0, 0, h], [0, w, h/2]]),  
#                                     'd':np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0  ]])},
#                           'planes':{'a':np.array([[1], [0]]), 
#                                     'b':np.array([[0], [1]]), 
#                                     'c':np.array([[0], [0]]), 
#                                     'd':np.array([[0], [0]])}
#                     }
            
#         if boundary == 'v':
#             self.max_val = tmax#700e-6#1#tmax#w/2
#             # x = 0, z = 0 & x = 0, z = h
#             # x = w, z = h/2
#             # t = 0
#             self.xtics = {'lines' :{'A':np.array([[0, 0, 0], [0, 0, h], [0, w, h/2]]), 
#                                     'd':np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0  ]])},
#                           'planes':{'a':np.array([[1]]), 
#                                     'b':np.array([[0]]), 
#                                     'c':np.array([[0]]), 
#                                     'd':np.array([[0]])}
#                          }

#         if boundary == 'xx':
#             self.max_val = w
#             # x = w
#             self.xtics = {'planes':{'a':np.array([[0]]), 
#                                     'b':np.array([[1]]), 
#                                     'c':np.array([[0]]), 
#                                     'd':np.array([[w]])}
#                          }
        
#         if boundary == 'yy':
#             self.max_val = h/2
#             # z = 0
#             # z = h
#             self.xtics = {'planes':{'a':np.array([[0], [0]]), 
#                                     'b':np.array([[0], [0]]), 
#                                     'c':np.array([[1], [1]]), 
#                                     'd':np.array([[0], [h]])}
#                          }
        
#         if boundary == 'xy':
#             self.max_val = h/2
#             # z = 0
#             # z = h
#             # x = 0
#             # x = w
#             self.xtics = {'planes':{'a':np.array([[0], [0], [0], [0]]), 
#                                     'b':np.array([[0], [0], [1], [1]]), 
#                                     'c':np.array([[1], [1], [0], [0]]), 
#                                     'd':np.array([[0], [h], [0], [w]])}
#                          }
            
#         self.boundary = boundary

#     def line_dist(self, t, x, z):
#         P = np.hstack((t, x, z))
#         A = self.xtics['lines']['A']
#         d = self.xtics['lines']['d']
#         AP = P- A
#         cross_product = np.cross(AP, d, axis =1)
#         cross_product_magnitude = np.linalg.norm(cross_product, axis=1)
#         direction_magnitude = np.linalg.norm(d, axis=1)

#         distance = cross_product_magnitude / direction_magnitude
#         # print(distance.shape)
#         distance = np.min(distance, keepdims=True)
#         # print(distance.shape)
        
#         return np.abs(distance)


#     def plane_dist(self, t, x, z):
#         a = self.xtics['planes']['a']
#         b = self.xtics['planes']['b']
#         c = self.xtics['planes']['c']
#         d = self.xtics['planes']['d']
#         num = np.abs(t*a + x*b + z*c - d)
#         den = np.sqrt(a**2 + b**2 + c**2)

#         distance = (num/den).reshape(-1, )
#         # print(distance.shape)
#         distance = np.min(distance, keepdims=True)
#         # print(distance.shape)

#         return np.abs(distance)

#     def eval(self, t, x, y):
#         if isinstance(x, (int, float, complex)):
#             x = np.array(x)
#         if isinstance(y, (int, float, complex)):
#             y = np.array(y)
#         if isinstance(t, (int, float, complex)):
#             t = t*np.ones_like(x)

#         t = t.reshape(-1, 1)
#         x = x.reshape(-1, 1)
#         y = y.reshape(-1, 1)
        
#         # A = np.hstack((t, x, y))
#         distances = []

#         key = list(self.xtics.keys())
#         if 'lines' in key:
#             distances.append(self.batch_line_dist(t, x, y))
#         if 'planes' in key:
#             distances.append(self.batch_plane_dist(t, x, y))

#         distances = np.hstack(distances)

#         distances = np.min(distances, axis=1, keepdims=True)
#         # print(distances.shape)
#         return  distances/self.max_val


