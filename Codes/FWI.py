import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from scipy.optimize import minimize

path_data_loading = ''
path_data_saving = ''

###------------------------Functions for grid generation--------------------###
###############################################################################
def StraightLineDiv(startp, endp, div):
    ""
    if startp[0]!=endp[0]:
        X = startp[0]*(1-div)+endp[0]*div
        Y = startp[1]
    else:
        X = startp[0]
        Y = startp[1]*(1-div)+endp[1]*div
    p=np.array([X, Y])
    return p;

def Coons(n_ksi, n_h):
    ""
    nele = n_ksi*n_h
    nnodes = (n_ksi+1)*(n_h+1)
    ele = np.zeros((nele,4),dtype=int)
    nodes = np.zeros((nnodes,2))
    for i in range(0, n_ksi):
        for j in range(0, n_h):
            ele[j*n_ksi+i,0]=(n_ksi+1)*j+i+1
            ele[j*n_ksi+i,1]=ele[j*n_ksi+i,0]+1;
            ele[j*n_ksi+i,2]=ele[j*n_ksi+i,1]+n_ksi+1;
            ele[j*n_ksi+i,3]=ele[j*n_ksi+i,2]-1;

            nodes[ele[j*n_ksi+i,0]-1,:]=[i/n_ksi, j/n_h]
            nodes[ele[j*n_ksi+i,1]-1,:]=[(i+1)/n_ksi, j/n_h]
            nodes[ele[j*n_ksi+i,2]-1,:]=[(i+1)/n_ksi, (j+1)/n_h]
            nodes[ele[j*n_ksi+i,3]-1,:]=[i/n_ksi, (j+1)/n_h]
    mesh={'elements':ele, 'nodesC':nodes}

    return mesh;

def structured_mesh_Rectangle(W, H, dx, dy):

    n_ksi = int(W / dx)
    
    n_h = int(H / dy)

    A=np.array([0, 0])
    B=np.array([W, 0])
    C=np.array([W, H])
    D=np.array([0, H])

    nodes = np.zeros(((n_ksi+1)*(n_h+1),2))

    mesh = Coons(n_ksi, n_h)
    ele = mesh['elements'] #Array to store each element
    nodesC = mesh['nodesC'] #Nodal coordinates

    nodes = np.zeros(((n_ksi + 1) * (n_h + 1), 2))

    for i in range(0, n_ksi*n_h):
        for j in range(0, 4):
            ksi=nodesC[ele[i,j]-1, 0]
            h=nodesC[ele[i,j]-1, 1]
            E0_ksi=1-ksi
            E1_ksi=ksi
            E0_h=1-h
            E1_h=h
            AB=StraightLineDiv(A, B, ksi)
            BC=StraightLineDiv(B, C, h)
            CD=StraightLineDiv(D, C, ksi)
            DA=StraightLineDiv(A, D, h)
            nodes[ele[i,j]-1,0]=E0_ksi*DA[0]+E1_ksi*BC[0]+E0_h*AB[0]+E1_h*CD[0]-E0_ksi*E0_h*A[0]-E1_ksi*E0_h*B[0]-E0_ksi*E1_h*D[0]-E1_ksi*E1_h*C[0]
            nodes[ele[i,j]-1,1]=E0_ksi*DA[1]+E1_ksi*BC[1]+E0_h*AB[1]+E1_h*CD[1]-E0_ksi*E0_h*A[1]-E1_ksi*E0_h*B[1]-E0_ksi*E1_h*D[1]-E1_ksi*E1_h*C[1]

    mesh['nodes']=nodes

    dx_real = W / n_ksi
    dy_real = H / n_h

    return n_ksi, n_h, dx_real, dy_real, mesh;
###############################################################################

###-----------------------Functions for pulse generation--------------------###
###############################################################################
def tukey_asymmetric(M, alpha_left=0.1, alpha_right=0.5):

    if M <= 1:
        return np.ones(M)

    aL = float(np.clip(alpha_left, 0.0, 1.0))
    aR = float(np.clip(alpha_right, 0.0, 1.0))

    n = np.arange(M)
    w = np.ones(M)

    L = int(np.floor(aL * (M - 1) / 2.0))
    R = int(np.floor(aR * (M - 1) / 2.0))

    if L > 0:
        nL = n[:L+1]
        w[:L+1] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * nL / (aL * (M - 1)) - 1.0)))

    if R > 0:
        nR = n[M-1-R:]
        w[M-1-R:] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * nR / (aR * (M - 1)) - 2.0 / aR + 1.0)))

    return w

def excitation_pulse(fc, Nc, t_sim, f_Amp):
    
    dt = t_sim[1]

    wf=2*np.pi*fc
    T=1/fc
    Tc=Nc*T
    
    g_T = np.sin(wf * t_sim[: int(Tc / dt)])
    
    ht = tukey_asymmetric(len(g_T), alpha_left=0.1, alpha_right=1.0)

    ft = np.zeros(int((73 * 4e-9) / dt))
    
    ft = np.append(ft, g_T * ht)
        
    ft = np.append(ft, np.zeros(t_sim.shape[0] - ft.shape[0]))

    ft = f_Amp * ft

    return ft;
###############################################################################

###----------------------------Phassed array specs--------------------------###
###############################################################################
N_element = 64
pitch_element = 1.0e-3
spacing_element = 0.3e-3
###############################################################################

###-----------------------------Sample dimensions---------------------------###
###############################################################################
W = 2.0 * (pitch_element * 9.0) + (N_element - 1.0) * pitch_element
H = 34.2e-3

print("DIMENSIONS")
print('The width is %.2f mm' %(W * 1e3))
print('The height is %.2f mm' %(H * 1e3))
###############################################################################

###-----------------------Background material properties--------------------###
###############################################################################
dens = 2680

cL = 6298.342541436465

cS = 3175.4874651810587 

cS_cL_ratio = cS / cL

v = ((1 / cS_cL_ratio) ** 2 - 2) / (2.0 * ((1 / cS_cL_ratio) ** 2 - 1.0))

E = (cL ** 2) * dens * (1 + v) * (1 - 2 * v) / (1 - v)

lamda = E * v / ((1 + v) * (1 - 2.0 * v))

mu = E / (2.0 * (1 + v))

beta = 5 * 1e8 / dens

print("MATERIAL PROPERTIES") 
print("The material density is %.2f kg/m^3" %(dens)) 
print("The lame parameters λ, μ are respectively %.2f GPa and %.2f GPa" %(lamda / 1e9, mu / 1e9)) 
print("The lognitudinal wave speed is %.2f m/s\n" %(cL))
print("The shear wave speed is %.2f m/s\n" %(cS))
###############################################################################

###-------------------Grid generation & Time discretization-----------------###
###############################################################################
fc=1e6 #Center frequency of incident wave in Hz

wavelength = cL / fc

dx = wavelength / 20.0
dy = wavelength / 20.0

dx = min(wavelength / 20.0, (pitch_element - spacing_element) / 4.0)
dy = min(wavelength / 20.0, (pitch_element - spacing_element) / 4.0)

n_ksi, n_h, dx_real, dy_real, mesh = structured_mesh_Rectangle(W, H, dx, dy)

t_one_pass = H / cL

dt_crit = min(dx_real, dy_real) / cL
dt = dt_crit / 1.5

start_source_pos_x = (W - (N_element - 1.0) * pitch_element) / 2.0

t1 = (np.sqrt((start_source_pos_x ** 2) + (H ** 2)) + np.sqrt(((W - start_source_pos_x) ** 2) + (H ** 2))) / cL

t2 = 3 * t_one_pass

t_end = max(t1, t2)
t_sim = np.linspace(0, t_end, int(t_end / dt + 1))
dt_real = t_sim[1]

print('SPATIAL AND TIME DISCRETIZATION')
print("The gridsize is %.2f mm" %(dx_real * 1e3))
print("The timestep is %.2f μs" %(dt_real * 1e6))
print("The time for one pass and the total simulaton time is %.2f μs and %.2f μs" %(t_one_pass * 1e6, t_end * 1e6))
###############################################################################

###------------------------Definition of source load------------------------###
###############################################################################
Nc = 6

f_Amp = 1.0

ft = excitation_pulse(fc, Nc, t_sim, f_Amp)
###############################################################################

###--------------------------Model of inverse solver------------------------###
###############################################################################
class Wave_equation_inverse_solver_2D:
  def __init__(self, n_ksi, n_h, material_density_CL, material_density_B, wavespeed_ratio_logit):
      
      self.n_ksi = n_ksi
      self.n_h = n_h

      self.material_density_CL = tf.Variable(material_density_CL, dtype = tf.float32, name = "material_density_CL")
      
      self.material_density_B = tf.Variable(material_density_B, dtype = tf.float32, name = "material_density_B")
      
      self.wavespeed_ratio_logit = tf.Variable(wavespeed_ratio_logit, dtype = tf.float32, name = "wave_speed_ratio_logit")

      return ;
      
  def Pi_matrix_assembly(self, source_pos):

      Pi = tf.zeros((self.n_h + 1, self.n_ksi), dtype=tf.float32)
    
      vals = [1.0, ] * len(source_pos)

      Pi = tf.tensor_scatter_nd_update(Pi, source_pos, vals)

      return Pi;
    
  def arithmetic_mean_assembly(self, ):

      self.arithmetic_mean = tf.reshape(tf.constant([[0.25, 0.25],
                                                     [0.25, 0.25]], dtype=tf.float32), [2, 2, 1, 1])

      return;
  
  @tf.function
  def one_time_step_forward_pass(self, inputs, ft_tensor_t, dx, dy, Pi, CS_squared, CL_squared_cell_centers, CL_squared_2CS_squared_diff_cell_centers, Beta_dt_x_inverse, Beta_dt_y_inverse, source_pos, recording_poss, dt, dt_squared):
      
      output_t, u_x_der_t, u_y_der_t, sxy_der_x_t, sxy_der_y_t, u_x_der_x_t, u_x_der_y_t, u_y_der_x_t, u_y_der_y_t, prev_hidden_x, prev_hidden_y = inputs
        
      xt = Pi * ft_tensor_t
      
      u_x_der_x_t = (prev_hidden_x[0, :, 1:] - prev_hidden_x[0, :, :-1]) / dx
      
      u_x_der_y_t = (prev_hidden_x[0, 1:, 1:-1] - prev_hidden_x[0, :-1, 1:-1]) / dy
      
      u_y_der_x_t =  (prev_hidden_y[0, 1:-1, 1:] - prev_hidden_y[0, 1:-1, :-1]) / dx
      
      u_y_der_y_t = (prev_hidden_y[0, 1:] - prev_hidden_y[0, :-1]) / dy
      
      ##########################################################################################################################
      sxx = CL_squared_cell_centers * u_x_der_x_t + CL_squared_2CS_squared_diff_cell_centers * u_y_der_y_t
    
      syy = CL_squared_2CS_squared_diff_cell_centers * u_x_der_x_t + CL_squared_cell_centers * u_y_der_y_t

      sxy_interior = CS_squared[1: -1, 1: -1] * (u_x_der_y_t + u_y_der_x_t)
      ##########################################################################################################################
      
      ##########################################################################################################################      
      sxx_padded = tf.concat([-sxx[:, 0:1],
                              sxx,
                              -sxx[:, -1:]], axis = 1)
        
      syy_padded = tf.concat([-syy[0:1],
                              syy,
                              -syy[-1:]], axis = 0)
                              
      sxy = tf.pad(sxy_interior, tf.constant([[1, 1], [1, 1]]), "CONSTANT", constant_values = 0.0)
      ##########################################################################################################################      
      
      ##########################################################################################################################
      sxy_der_x_t = (sxy[:, 1:] - sxy[:, :-1]) / dx
      
      sxy_der_y_t = (sxy[1:] - sxy[:-1]) / dy
      ##########################################################################################################################
    
      ##########################################################################################################################
      next_hidden_x = (dt_squared * ((sxx_padded[:, 1:] - sxx_padded[:, :-1]) / dx + sxy_der_y_t) + \
                      (prev_hidden_x[0] - prev_hidden_x[1]) * 2.0) * Beta_dt_x_inverse + prev_hidden_x[1]
    
      next_hidden_y = (dt_squared * (sxy_der_x_t + (syy_padded[1:] - syy_padded[:-1]) / dy) + \
                      (prev_hidden_y[0] - prev_hidden_y[1]) * 2.0) * Beta_dt_y_inverse + prev_hidden_y[1]  + xt
      ##########################################################################################################################
      
      u_x_der_t = (next_hidden_x - prev_hidden_x[1]) / (2.0 * dt)
      
      u_y_der_t = (next_hidden_y - prev_hidden_y[1]) / (2.0 * dt)
      
      output_t = tf.gather_nd(prev_hidden_y[0], recording_poss)
      
      ##########################################################################################################################
      prev_hidden_x = tf.stack([next_hidden_x, prev_hidden_x[0]], axis=0)
    
      prev_hidden_y = tf.stack([next_hidden_y, prev_hidden_y[0]], axis=0)        
      ##########################################################################################################################
    
      return output_t, u_x_der_t, u_y_der_t, sxy_der_x_t, sxy_der_y_t, u_x_der_x_t, u_x_der_y_t, u_y_der_x_t, u_y_der_y_t, prev_hidden_x, prev_hidden_y;
  
  @tf.function
  def forward_pass(self, dx, dy, ft_tensor, CS_squared, CL_squared_cell_centers, CL_squared_2CS_squared_diff_cell_centers, Beta_dt_x_inverse, Beta_dt_y_inverse, source_pos, recording_poss, dt, dt_squared):
      
      Pi = self.Pi_matrix_assembly(source_pos)
      
      output_t_0 = tf.zeros(len(recording_poss), dtype=tf.float32)

      prev_hidden_x_0 = tf.zeros((2, self.n_h, self.n_ksi + 1), dtype=tf.float32)

      prev_hidden_y_0 = tf.zeros((2, self.n_h + 1, self.n_ksi), dtype=tf.float32)
      
      u_x_der_t_0 = tf.zeros((self.n_h, self.n_ksi + 1), dtype=tf.float32)
      
      u_y_der_t_0 = tf.zeros((self.n_h + 1, self.n_ksi), dtype=tf.float32)
      
      sxy_der_x_t_0 = tf.zeros((self.n_h + 1, self.n_ksi), dtype=tf.float32)
      
      sxy_der_y_t_0 = tf.zeros((self.n_h, self.n_ksi + 1), dtype=tf.float32)
      
      u_x_der_x_t_0 = tf.zeros((self.n_h, self.n_ksi), dtype=tf.float32)
      
      u_x_der_y_t_0 = tf.zeros((self.n_h - 1, self.n_ksi - 1), dtype=tf.float32)
      
      u_y_der_x_t_0 = tf.zeros((self.n_h - 1, self.n_ksi - 1), dtype=tf.float32)
      
      u_y_der_y_t_0 = tf.zeros((self.n_h, self.n_ksi), dtype=tf.float32)

      output, u_x_der_t, u_y_der_t, sxy_der_x_t, sxy_der_y_t, u_x_der_x_t, u_x_der_y_t, u_y_der_x_t, u_y_der_y_t = tf.scan(fn = lambda inputs, ft_tensor_t: self.one_time_step_forward_pass(inputs, ft_tensor_t, dx, dy, Pi, CS_squared, CL_squared_cell_centers, CL_squared_2CS_squared_diff_cell_centers, Beta_dt_x_inverse, Beta_dt_y_inverse, source_pos, recording_poss, dt, dt_squared),
                                                                                                                           elems = ft_tensor,
                                                                                                                           initializer = (output_t_0, u_x_der_t_0, u_y_der_t_0, sxy_der_x_t_0, sxy_der_y_t_0, u_x_der_x_t_0, u_x_der_y_t_0, u_y_der_x_t_0, u_y_der_y_t_0, prev_hidden_x_0, prev_hidden_y_0))[:-2]
                                                         
      output = tf.transpose(output)

      return output, u_x_der_t, u_y_der_t, sxy_der_x_t, sxy_der_y_t, u_x_der_x_t, u_x_der_y_t, u_y_der_x_t, u_y_der_y_t;
      
  @tf.function
  def one_time_step_backward_pass(self, inputs, adjoint_ft_tensor_t, dx, dy, CS_squared, CL_squared_cell_centers, CL_squared_2CS_squared_diff_cell_centers, Beta_dt_x_inverse, Beta_dt_y_inverse, source_pos, recording_poss, dt_squared):
      
      next_hidden_x, next_hidden_y = inputs
      
      xt = tf.scatter_nd(recording_poss, adjoint_ft_tensor_t, shape=(self.n_h + 1, self.n_ksi))
      
      ##########################################################################################################################
      sxx = CL_squared_cell_centers * (next_hidden_x[0, :, 1:] - next_hidden_x[0, :, :-1]) / dx + CL_squared_2CS_squared_diff_cell_centers * (next_hidden_y[0, 1:] - next_hidden_y[0, :-1]) / dy
    
      syy = CL_squared_2CS_squared_diff_cell_centers * (next_hidden_x[0, :, 1:] - next_hidden_x[0, :, :-1]) / dx + CL_squared_cell_centers * (next_hidden_y[0, 1:] - next_hidden_y[0, :-1]) / dy  

      sxy_interior = CS_squared[1: -1, 1: -1] * ((next_hidden_x[0, 1:, 1:-1] - next_hidden_x[0, :-1, 1:-1]) / dy + (next_hidden_y[0, 1:-1, 1:] - next_hidden_y[0, 1:-1, :-1]) / dx)
      ##########################################################################################################################
      
      ##########################################################################################################################      
      sxx_padded = tf.concat([-sxx[:, 0:1],
                              sxx,
                              -sxx[:, -1:]], axis = 1)
        
      syy_padded = tf.concat([-syy[0:1],
                              syy,
                              -syy[-1:]], axis = 0)
                              
      sxy = tf.pad(sxy_interior, tf.constant([[1, 1], [1, 1]]), "CONSTANT", constant_values = 0.0)
      ##########################################################################################################################      
    
      ##########################################################################################################################
      prev_hidden_x = (dt_squared * ((sxx_padded[:, 1:] - sxx_padded[:, :-1]) / dx + (sxy[1:] - sxy[:-1]) / dy) + \
                      (next_hidden_x[0] - next_hidden_x[1]) * 2.0) * Beta_dt_x_inverse + next_hidden_x[1]
    
      prev_hidden_y = (dt_squared * ((sxy[:, 1:] - sxy[:, :-1]) / dx + (syy_padded[1:] - syy_padded[:-1]) / dy) + \
                      (next_hidden_y[0] - next_hidden_y[1]) * 2.0) * Beta_dt_y_inverse + next_hidden_y[1]  + xt      
      ##########################################################################################################################

      ##########################################################################################################################    
      next_hidden_x = tf.stack([prev_hidden_x, next_hidden_x[0]], axis=0)
    
      next_hidden_y = tf.stack([prev_hidden_y, next_hidden_y[0]], axis=0)
      ##########################################################################################################################
    
      return next_hidden_x, next_hidden_y;      
      
  @tf.function
  def backward_pass(self, dx, dy, adjoint_ft_tensor, CS_squared, CL_squared_cell_centers, CL_squared_2CS_squared_diff_cell_centers, Beta_dt_x_inverse, Beta_dt_y_inverse, source_pos, recording_poss, dt_squared):

      next_hidden_x_0 = tf.zeros((2, self.n_h, self.n_ksi + 1), dtype=tf.float32)

      next_hidden_y_0 = tf.zeros((2, self.n_h + 1, self.n_ksi), dtype=tf.float32)

      next_hidden_x, next_hidden_y = tf.scan(fn = lambda inputs, adjoint_ft_tensor_t: self.one_time_step_backward_pass(inputs, adjoint_ft_tensor_t, dx, dy, CS_squared, CL_squared_cell_centers, CL_squared_2CS_squared_diff_cell_centers, Beta_dt_x_inverse, Beta_dt_y_inverse, source_pos, recording_poss, dt_squared),
                                             elems = adjoint_ft_tensor,
                                             initializer = (next_hidden_x_0, next_hidden_y_0))

      return next_hidden_x, next_hidden_y;
    
  @tf.function
  def loop_body(self, inputs, dx, dy, ft_tensor, wavespeed_ratio_squared, cL_squared, cS_squared, cL_squared_2cS_squared_diff, CS_squared, CL_squared_cell_centers, CL_squared_2CS_squared_diff_cell_centers, beta, Beta_dt_x_inverse, Beta_dt_y_inverse, recording_poss, dt, dt_squared):
      
      source_pos, ground_truth_output, ground_truth_output_mean = inputs
      
      output, u_x_der_t, u_y_der_t, sxy_der_x_t, sxy_der_y_t, u_x_der_x_t, u_x_der_y_t, u_y_der_x_t, u_y_der_y_t  = self.forward_pass(dx, dy, ft_tensor, CS_squared, CL_squared_cell_centers, CL_squared_2CS_squared_diff_cell_centers, Beta_dt_x_inverse, Beta_dt_y_inverse, source_pos, recording_poss, dt, dt_squared)
      
      output = output - ground_truth_output_mean
      
      res = 2.0 * ((1.0 / len(recording_poss)) ** 2) * (output - ground_truth_output)
      
      adjoint_ft_tensor = tf.reverse(tf.transpose(res), axis = [0]) * dt_squared
      
      adjoint_next_hidden_x, adjoint_next_hidden_y = self.backward_pass(dx, dy, adjoint_ft_tensor, CS_squared, CL_squared_cell_centers, CL_squared_2CS_squared_diff_cell_centers, Beta_dt_x_inverse, Beta_dt_y_inverse, source_pos, recording_poss, dt_squared)
      
      adjoint_u_x = tf.reverse(adjoint_next_hidden_x[:, 1], axis = [0])
      
      adjoint_u_y = tf.reverse(adjoint_next_hidden_y[:, 1], axis = [0])
      
      gradients_material_density_CL = -(tf.squeeze(tf.nn.conv2d(tf.reshape(tf.reduce_sum((adjoint_u_x[:, :, 1:] - adjoint_u_x[:, :, :-1]) * (cL_squared * u_x_der_x_t + cL_squared_2cS_squared_diff * u_y_der_y_t) / dx + \
                                                                                       (adjoint_u_y[:, 1:] - adjoint_u_y[:, :-1]) * (cL_squared_2cS_squared_diff * u_x_der_x_t + cL_squared * u_y_der_y_t) / dy, axis = 0), [1, self.n_h, self.n_ksi, 1]), self.arithmetic_mean, strides=[1, 1, 1, 1], padding="VALID")) + \
                                        tf.reduce_sum(((adjoint_u_x[:, 1:, 1: -1] - adjoint_u_x[:, :-1, 1: -1]) / dy + (adjoint_u_y[:, 1: -1, 1:] - adjoint_u_y[:, 1: -1, :-1]) / dx) * cS_squared * (u_x_der_y_t + u_y_der_x_t), axis = 0))

      Bx = (u_x_der_t * adjoint_u_x)[:, :, 1: -1]
      By = (u_y_der_t * adjoint_u_y)[:, 1: -1]
      gradients_material_density_B = -beta * 0.5 * tf.reduce_sum(Bx[:, 1:] + Bx[:, :-1] + By[:, :, 1:] + By[:, :, :-1], axis = 0)
     
      gradients_wavespeed_ratio_logit = (tf.reduce_sum(adjoint_u_x * sxy_der_y_t) + tf.reduce_sum(adjoint_u_y * sxy_der_x_t)) / wavespeed_ratio_squared
      
      loss = (1 / len(recording_poss)) * tf.reduce_sum(tf.square(output - ground_truth_output))
      
      return loss, gradients_material_density_CL, gradients_material_density_B, gradients_wavespeed_ratio_logit;
  
  @tf.function
  def gradients_calc(self, ft_tensor, dt, dt_squared, cL_squared, cL_dt_squared, beta, beta_dt, source_pos_list, recording_poss, ground_truth_output_list, ground_truth_output_mean_list, w):
      
      material_density_CL_post_activation = 1.0 / (tf.square(tf.pad(self.material_density_CL, tf.constant([[1, 1], [1, 1]]), "CONSTANT", constant_values = 0.0)) + 1.0)
      
      material_density_B_post_activation = tf.square(tf.pad(self.material_density_B, tf.constant([[1, 1], [1, 1]]), "CONSTANT", constant_values = 0.0)) + 1.0
          
      wavespeed_ratio_squared = tf.square(tf.sigmoid(self.wavespeed_ratio_logit))
      
      cS_squared = wavespeed_ratio_squared * cL_squared
      
      cL_squared_2cS_squared_diff = cL_squared -  2.0 * cS_squared
      
      material_density_CL_post_activation_cell_centers = tf.squeeze(tf.nn.conv2d(tf.reshape(material_density_CL_post_activation, [1, self.n_h + 1, self.n_ksi + 1, 1]), self.arithmetic_mean, strides=[1, 1, 1, 1], padding="VALID"))
          
      CL_squared_cell_centers = material_density_CL_post_activation_cell_centers * cL_squared
      
      CS_squared = material_density_CL_post_activation * cS_squared
      
      CL_squared_2CS_squared_diff_cell_centers = material_density_CL_post_activation_cell_centers * cL_squared_2cS_squared_diff
      
      Beta_dt_x_inverse = 1.0 / (1.0 + 0.5 * beta_dt * 0.5 * (material_density_B_post_activation[:-1] + material_density_B_post_activation[1:]))
      
      Beta_dt_y_inverse = 1.0 / (1.0 + 0.5 * beta_dt * 0.5 * (material_density_B_post_activation[:, :-1] + material_density_B_post_activation[:, 1:]))
          
      loss, gradients_material_density_CL, gradients_material_density_B, gradients_wavespeed_ratio_logit = tf.map_fn(fn = lambda inputs: self.loop_body(inputs, dx, dy, ft_tensor, wavespeed_ratio_squared, cL_squared, cS_squared, cL_squared_2cS_squared_diff, CS_squared, CL_squared_cell_centers, CL_squared_2CS_squared_diff_cell_centers, beta, Beta_dt_x_inverse, Beta_dt_y_inverse, recording_poss, dt, dt_squared), 
                                                                                                                     elems = (source_pos_list, ground_truth_output_list, ground_truth_output_mean_list), dtype=(tf.float32, tf.float32, tf.float32, tf.float32))  
    
      pearson_reg_numerator = tf.reduce_mean((material_density_CL_post_activation - tf.reduce_mean(material_density_CL_post_activation)) * (material_density_B_post_activation - tf.reduce_mean(material_density_B_post_activation)))
                       
      pearson_reg_denominator = tf.math.reduce_std(material_density_CL_post_activation) * tf.math.reduce_std(material_density_B_post_activation)           
 
      loss_tot = tf.reduce_mean(loss) + w * tf.square((pearson_reg_numerator / pearson_reg_denominator) + 1.0)
      

      gradients_material_density_CL = tf.reduce_sum(gradients_material_density_CL, axis = 0) + (w * 2.0 * (((pearson_reg_numerator / pearson_reg_denominator) + 1.0) / ((self.n_h + 1) * (self.n_ksi + 1) * pearson_reg_denominator)) * ((material_density_B_post_activation - tf.reduce_mean(material_density_B_post_activation)) - (pearson_reg_numerator / pearson_reg_denominator) * (tf.math.reduce_std(material_density_B_post_activation) / tf.math.reduce_std(material_density_CL_post_activation)) * (material_density_CL_post_activation - tf.reduce_mean(material_density_CL_post_activation))))[1: -1, 1: -1]
      gradients_material_density_CL = gradients_material_density_CL * (-2.0 * self.material_density_CL) / tf.square((tf.square(self.material_density_CL) + 1.0))    


      gradients_material_density_B = tf.reduce_sum(gradients_material_density_B, axis = 0) + (w * 2.0 * (((pearson_reg_numerator / pearson_reg_denominator) + 1.0) / ((self.n_h + 1) * (self.n_ksi + 1) * pearson_reg_denominator)) * ((material_density_CL_post_activation - tf.reduce_mean(material_density_CL_post_activation)) - (pearson_reg_numerator / pearson_reg_denominator) * (tf.math.reduce_std(material_density_CL_post_activation) / tf.math.reduce_std(material_density_B_post_activation)) * (material_density_B_post_activation - tf.reduce_mean(material_density_B_post_activation))))[1: -1, 1: -1]
      gradients_material_density_B = gradients_material_density_B * 2.0 * self.material_density_B
          
      gradients_wavespeed_ratio_logit = tf.reduce_sum(gradients_wavespeed_ratio_logit, axis = 0) * 2.0 * tf.square(tf.sigmoid(self.wavespeed_ratio_logit)) * (1.0 - tf.sigmoid(self.wavespeed_ratio_logit))
          
      gradients = [gradients_material_density_CL, gradients_material_density_B, gradients_wavespeed_ratio_logit]
      
      return gradients, loss_tot;
      
  def gradients_calc_for_optimizer(self, params, ft_tensor, dt, dt_squared, cL_squared, cL_dt_squared, beta, beta_dt, source_pos_list, recording_poss, ground_truth_output_list, ground_truth_output_mean_list):
      
      if len(params) == 1:
      
          self.wavespeed_ratio_logit.assign(tf.cast(params[0], dtype = tf.float32))
          
          w = 0
    
      elif len(params) == 2 * (self.n_h - 1) * (self.n_ksi - 1):
      
          self.material_density_CL.assign(tf.reshape(tf.cast(params[:(self.n_h - 1) * (self.n_ksi - 1)], dtype = tf.float32), [self.n_h - 1, self.n_ksi - 1]))
      
          self.material_density_B.assign(tf.reshape(tf.cast(params[(self.n_h - 1) * (self.n_ksi - 1):], dtype = tf.float32), [self.n_h - 1, self.n_ksi - 1]))
         
          w = 0.1
         
      else:
      
          self.material_density_CL.assign(tf.reshape(tf.cast(params[:(self.n_h - 1) * (self.n_ksi - 1)], dtype = tf.float32), [self.n_h - 1, self.n_ksi - 1]))
      
          self.material_density_B.assign(tf.reshape(tf.cast(params[(self.n_h - 1) * (self.n_ksi - 1):-1], dtype = tf.float32), [self.n_h - 1, self.n_ksi - 1]))
  
          self.wavespeed_ratio_logit.assign(tf.cast(params[-1], dtype = tf.float32))      

          w = 0.1
  
      gradients, loss_tot = self.gradients_calc(ft_tensor, dt, dt_squared, cL_squared, cL_dt_squared, beta, beta_dt, source_pos_list, recording_poss, ground_truth_output_list, ground_truth_output_mean_list, w)
      
      self.loss_current = loss_tot
      
      if len(params) == 1:
      
          gradients = tf.cast(gradients[2], dtype = tf.float64).numpy().flatten()
      
      
      elif len(params) == 2 * (self.n_h - 1) * (self.n_ksi - 1):
      
          gradients = np.concatenate([tf.cast(gradients[0], dtype = tf.float64).numpy().flatten() , tf.cast(gradients[1], dtype = tf.float64).numpy().flatten()])
      
      
      else:
      
          gradients = np.concatenate([tf.cast(gradients[0], dtype = tf.float64).numpy().flatten() , tf.cast(gradients[1], dtype = tf.float64).numpy().flatten(), tf.cast(gradients[2], dtype = tf.float64).numpy().flatten()])
  
      return gradients, tf.cast(loss_tot, dtype = tf.float64).numpy();

  def solver(self, dx, dy, dt, ft, cL, beta, source_pos_list, recording_poss, training_epochs, ground_truth_output_list, ground_truth_output_mean_list, path_data_saving):
      
      dx = tf.constant(dx, dtype = tf.float32)        
      dy = tf.constant(dy, dtype = tf.float32)
      
      dt = tf.constant(dt, dtype = tf.float32)
      
      dt_squared = tf.square(dt)
      
      cL_squared = tf.constant(cL ** 2, dtype = tf.float32)

      cL_dt_squared = tf.constant(cL_squared * dt_squared, dtype = tf.float32)
      
      beta = tf.constant(beta, dtype = tf.float32)
      
      beta_dt = beta * dt
      
      ft_tensor = tf.constant(ft.reshape(-1, 1), dtype = tf.float32)
      
      self.arithmetic_mean_assembly()
      
      initial_params = np.concatenate([tf.cast(self.material_density_CL, dtype = tf.float64).numpy().flatten(), tf.cast(self.material_density_B, dtype = tf.float64).numpy().flatten(), tf.cast(self.wavespeed_ratio_logit, dtype = tf.float64).numpy().flatten()])
      
      self.flag = 0
      
      self.training_epoch = 0
      
      self.training_epochs = training_epochs
      
      self.loss_current = 0
      
      self.loss_tot_list = []
      
      self.path_data_saving = path_data_saving
      
      np.savetxt("{}material_density_CL_%d.txt".format(self.path_data_saving) %(self.training_epoch), self.material_density_CL)
      
      np.savetxt("{}material_density_B_%d.txt".format(self.path_data_saving) %(self.training_epoch), self.material_density_B)
      
      np.savetxt("{}wavespeed_ratio_logit_%d.txt".format(self.path_data_saving) %(self.training_epoch), np.array([self.wavespeed_ratio_logit]))
      
      def callback(params_t):
      
          self.loss_tot_list.append(self.loss_current)
      
          tf.print("Training epoch: %d/%d, Loss: %.10f" %(self.training_epoch + 1, self.training_epochs, self.loss_tot_list[self.training_epoch]))
          
          self.training_epoch = self.training_epoch + 1
      
          np.savetxt("{}material_density_CL_%d.txt".format(self.path_data_saving) %(self.training_epoch), self.material_density_CL)
          
          np.savetxt("{}material_density_B_%d.txt".format(self.path_data_saving) %(self.training_epoch), self.material_density_B)
          
          np.savetxt("{}wavespeed_ratio_logit_%d.txt".format(self.path_data_saving) %(self.training_epoch), np.array([self.wavespeed_ratio_logit]))
      
          return ;
          
      initial_params = np.concatenate([tf.cast(self.wavespeed_ratio_logit, dtype = tf.float64).numpy().flatten(), ])
      
      minimize(lambda params: self.gradients_calc_for_optimizer(params, ft_tensor, dt, dt_squared, cL_squared, cL_dt_squared, beta, beta_dt, source_pos_list, recording_poss, ground_truth_output_list, ground_truth_output_mean_list)[1], 
               initial_params, 
               method='L-BFGS-B', 
               jac=lambda params: self.gradients_calc_for_optimizer(params, ft_tensor, dt, dt_squared, cL_squared, cL_dt_squared, beta, beta_dt, source_pos_list, recording_poss, ground_truth_output_list, ground_truth_output_mean_list)[0],  # Pass gradients
               options={'maxiter': training_epochs,
                        'ftol':0,
                        'disp': True},
               callback=callback)
               
      self.training_epoch_first_phase = self.training_epoch
      
      tf.print("The first phase concluded at training epoch %d/%d" %(self.training_epoch_first_phase, training_epochs))
      
      initial_params = np.concatenate([tf.cast(self.material_density_CL, dtype = tf.float64).numpy().flatten(), tf.cast(self.material_density_B, dtype = tf.float64).numpy().flatten()])
      
      minimize(lambda params: self.gradients_calc_for_optimizer(params, ft_tensor, dt, dt_squared, cL_squared, cL_dt_squared, beta, beta_dt, source_pos_list, recording_poss, ground_truth_output_list, ground_truth_output_mean_list)[1], 
               initial_params, 
               method='L-BFGS-B', 
               jac=lambda params: self.gradients_calc_for_optimizer(params, ft_tensor, dt, dt_squared, cL_squared, cL_dt_squared, beta, beta_dt, source_pos_list, recording_poss, ground_truth_output_list, ground_truth_output_mean_list)[0],  # Pass gradients
               options={'maxiter': 250,
                        'ftol':0,
                        'disp': True},
               callback=callback)     

      self.training_epoch_second_phase = self.training_epoch
      
      tf.print("The second phase concluded at training epoch %d/%d" %(self.training_epoch_second_phase, training_epochs))
      
      initial_params = np.concatenate([tf.cast(self.material_density_CL, dtype = tf.float64).numpy().flatten(), tf.cast(self.material_density_B, dtype = tf.float64).numpy().flatten(), tf.cast(self.wavespeed_ratio_logit, dtype = tf.float64).numpy().flatten()])
  
      minimize(lambda params: self.gradients_calc_for_optimizer(params, ft_tensor, dt, dt_squared, cL_squared, cL_dt_squared, beta, beta_dt, source_pos_list, recording_poss, ground_truth_output_list, ground_truth_output_mean_list)[1], 
               initial_params, 
               method='L-BFGS-B', 
               jac=lambda params: self.gradients_calc_for_optimizer(params, ft_tensor, dt, dt_squared, cL_squared, cL_dt_squared, beta, beta_dt, source_pos_list, recording_poss, ground_truth_output_list, ground_truth_output_mean_list)[0],  # Pass gradients
               options={'maxiter': training_epochs - self.training_epoch_second_phase,
                        'ftol':0,
                        'disp': True},
               callback=callback)      
               
      np.savetxt("{}loss_evolution.txt".format(self.path_data_saving), np.array(self.loss_tot_list))
      
      return ;
###############################################################################

###---------------------------Model initialization--------------------------###
###############################################################################
material_density_CL = np.random.normal(loc = 0.0, scale = 0.1, size = (n_h - 1, n_ksi - 1))

material_density_B = np.random.normal(loc = 0.0, scale = 0.1, size = (n_h - 1, n_ksi - 1))

wavespeed_ratio_logit = 0.0

model = Wave_equation_inverse_solver_2D(n_ksi, n_h, material_density_CL, material_density_B, wavespeed_ratio_logit)
###############################################################################

###-------------------------Selection of source points----------------------###
###############################################################################
source_pos_x = []
source_pos_y = H

start_source_pos_x = (W - (N_element - 1.0) * pitch_element) / 2.0

for i in range(0, N_element, 1):
    source_pos_x.append(start_source_pos_x + i * pitch_element)
    
source_pos_list = tf.convert_to_tensor([[[int(np.rint(source_pos_y / (H / n_h))), int(np.rint(source_pos_x[i] / (W / n_ksi))) + j] for j in range(-2, 3)] for i in range(len(source_pos_x))], dtype = tf.int32)
###############################################################################

###------------------------Selection of recording points--------------------###
###############################################################################
recording_poss_x = []
recording_poss_y = H

start_recording_poss_x = (W - (N_element - 1.0) * pitch_element) / 2.0

for i in range(0, N_element, 1):
    recording_poss_x.append(start_recording_poss_x + i * pitch_element)
    
recording_poss = tf.convert_to_tensor([[int(np.rint(recording_poss_y / (H / n_h))), int(np.rint(recording_poss_x[i] / (W / n_ksi)))] for i in range(len(recording_poss_x))], dtype = tf.int32)
###############################################################################

###--------------------------Loading of ground truth------------------------###
###############################################################################
ground_truth_output_list = []
ground_truth_output_mean_list = []

for i in range(len(source_pos_list)):
    
    ground_truth_output = np.loadtxt('{}U_%d_recorded_ground_truth_circular_defect_filled2_64.txt'.format(path_data_loading) %(i)).tolist()
    ground_truth_output_list.append(ground_truth_output)
    
    ground_truth_output_mean = np.loadtxt('{}U_mean_%d_recorded_ground_truth_circular_defect_filled2_64.txt'.format(path_data_loading) %(i)).tolist()
    ground_truth_output_mean_list.append(ground_truth_output_mean)
    
ground_truth_output_list = tf.convert_to_tensor(ground_truth_output_list, dtype=tf.float32)

ground_truth_output_mean_list = tf.convert_to_tensor(ground_truth_output_mean_list, dtype=tf.float32)
###############################################################################

###-------------------------------Run inversion-----------------------------###
###############################################################################
training_epochs = 400

model.solver(dx_real, dy_real, dt_real, ft, cL, beta, source_pos_list, recording_poss, training_epochs, ground_truth_output_list, ground_truth_output_mean_list, path_data_saving)
###############################################################################
