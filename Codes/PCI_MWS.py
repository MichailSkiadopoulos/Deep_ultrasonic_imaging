import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.cm as cm1
from matplotlib.ticker import FormatStrFormatter
import time

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

def unitSquareDiv(n_ksi, n_h):
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

##----------------------------Sample Dimensions--------------------------------
###############################################################################
H_sample = 34.2e-3
W_sample = 81e-3
###############################################################################

##----------------------------Pulse Parameters---------------------------------
###############################################################################
f=1e6
###############################################################################

##-----------------------Material Properties for sample------------------------
###############################################################################
cp_sample = 6298.342541436465

wavelength = cp_sample / f
###############################################################################

##--------------------------Defect dimensions and position---------------------
###############################################################################
r_defect = wavelength / 2.0
x_defect = W_sample / 2.0 + 1.0 * wavelength
y_defect = H_sample / 2.0 - wavelength / 2.0
###############################################################################

##-------------------------Phased array Dimensions-----------------------------
###############################################################################
N_element = 64
pitch_element = 1.0e-3
spacing_element = 0.3e-3
###############################################################################

##-----------------------------Grid parameters---------------------------------
###############################################################################
dx = wavelength / 20.0
dy = wavelength / 20.0

dx = min(wavelength / 20.0, (pitch_element - spacing_element) / 4.0)
dy = min(wavelength / 20.0, (pitch_element - spacing_element) / 4.0)
###############################################################################

n_ksi, n_h, dx_real, dy_real, mesh = structured_mesh_Rectangle(W_sample, H_sample, dx, dy)

##--------------------Partition for loading and recording----------------------
###############################################################################
x_trans = []
y_trans = np.zeros(len(range(0, N_element, 1)), )

start_recording_poss_x = (W_sample - (N_element - 1.0) * pitch_element) / 2.0

for i in range(0, N_element, 1):
    x_trans.append(start_recording_poss_x + i * pitch_element)
    
x_trans = np.array(x_trans)
    
x_trans = x_trans - W_sample / 2.0

x_tr = x_trans.copy()
y_tr = y_trans.copy()

x_rc = x_trans.copy()
y_rc = y_trans.copy()
###############################################################################

nodes = mesh['nodes'].copy()
nodes[:, 0] = nodes[:, 0] - W_sample / 2.0
nodes[:, 1] = nodes[:, 1] - H_sample

scanType = "Area" # "Area" or "Points"

index_nodes_interest = []

if scanType == "Area":
    
    xMin = -W_sample / 2 - dx_real / 10
    xMax = W_sample / 2 + dx_real / 10

    yMin = -H_sample - dy_real / 10
    yMax = dy_real / 10
    
    for i in range(nodes.shape[0]):
        if nodes[i, 0] >= xMin and nodes[i, 0] <= xMax and nodes[i, 1] >= yMin and nodes[i, 1] <= yMax:
            index_nodes_interest.append(i)
            
elif scanType == "Points":
    
    x_points = [3 * wavelength, ]
    y_points = [-H_sample / 2.0 - wavelength, ]
    
    for x_point, y_point in zip(x_points, y_points):
        dist = np.sqrt((x_point - nodes[:, 0]) ** 2 + (y_point - nodes[:, 1]) ** 2)
        
        index_nodes_interest.append(np.argmin(dist))
        
nodes_interest_x = nodes[index_nodes_interest, 0]
nodes_interest_y = nodes[index_nodes_interest, 1]

time_start = time.time()

d_tr = np.sqrt((x_tr[:, None] - nodes_interest_x[None, :]) ** 2 + (y_tr[:, None] - nodes_interest_y[None, :]) ** 2)

d_rc = np.sqrt((x_rc[:, None] - nodes_interest_x[None, :]) ** 2 + (y_rc[:, None] - nodes_interest_y[None, :]) ** 2)

t_tr_rc = (d_tr[:, None, :] + d_rc[None, :, :]) / cp_sample

path = ""

time_data = np.tile(np.loadtxt("{}simulation_time.txt".format(path)), (len(range(0, N_element, 1)) ** 2, 1))

FMC_data = []

for i in range(len(range(0, N_element, 1))):
    
    FMC_data.append(np.loadtxt('{}U_full_%d_recorded_ground_truth_circular_defect_filled2_64.txt.txt'.format(path) %(i)))
    
FMC_data = np.concatenate(FMC_data, axis = 0)

SAW_groups_signals_index = {}

for i in range(len(range(0, N_element, 1))):
    SAW_groups_signals_index[i] = []

for i in range(len(range(0, N_element, 1))):
    
    for j in range(len(range(0, N_element, 1))):
        
        count = i * len(range(0, N_element, 1)) + j
        
        SAW_groups_signals_index[abs(i - j)].append(count)
    
FMC_data_SAW_suppressed = np.zeros((len(range(0, N_element, 1)) ** 2, time_data.shape[1]))

for i in range(len(range(0, N_element, 1))):
    
    FMC_data_SAW_suppressed[SAW_groups_signals_index[i]] = FMC_data[SAW_groups_signals_index[i]] - np.mean(FMC_data[SAW_groups_signals_index[i]], axis = 0)
    
I_global = []
I_global_global = []

VCF_global_Re = []

VCF_global_Im = []

for i in range(len(range(0, N_element, 1))):
    
    print("Processing data from transmitter element %d" %(i + 1))
    
    I_local = np.zeros(nodes.shape[0], )
    I_local_local = []
    
    VCF_local_Re = np.zeros(nodes.shape[0], )
    
    VCF_local_Im = np.zeros(nodes.shape[0], )
    
    for j in range(len(range(0, N_element, 1))):
        
        count = i * len(range(0, N_element, 1)) + j
    
        time_signal = time_data[count]
        
        U2_signal = FMC_data_SAW_suppressed[count]
        
        analytic_U2_signal = hilbert(U2_signal)
            
        A = np.interp(t_tr_rc[i, j], time_signal, U2_signal)
        
        B = np.interp(t_tr_rc[i, j], time_signal, analytic_U2_signal)

        I_local[index_nodes_interest] = I_local[index_nodes_interest] + A
        
        VCF_local_Re[index_nodes_interest] = VCF_local_Re[index_nodes_interest] + np.real(B) / np.abs(B)
        
        VCF_local_Im[index_nodes_interest] = VCF_local_Im[index_nodes_interest] + np.imag(B) / np.abs(B)
        
        I_local_local.append(A)

    I_global.append(I_local)
    
    VCF_global_Re.append(VCF_local_Re)
    
    VCF_global_Im.append(VCF_local_Im)
    
    I_global_global.append(I_local_local)
    
time_taken = time.time() - time_start

print("The computational time is %f" %(time_taken))

I_global_final = np.zeros(mesh['nodes'].shape[0], )

VCF_global_final_Re = np.zeros(mesh['nodes'].shape[0], )

VCF_global_final_Im = np.zeros(mesh['nodes'].shape[0], )

for I in I_global:
    I_global_final = I_global_final + I
    
for VCF_Re in VCF_global_Re:
    VCF_global_final_Re = VCF_global_final_Re + VCF_Re
    
for VCF_Im in VCF_global_Im:
    VCF_global_final_Im = VCF_global_final_Im + VCF_Im
    
VCF_global_final = np.sqrt(VCF_global_final_Re ** 2 + VCF_global_final_Im ** 2)
    
I_VCF_global_final = I_global_final * VCF_global_final
    
I_global_final = np.abs(I_global_final)

I_VCF_global_final = I_global_final * VCF_global_final

fig = plt.figure(0, figsize = (40, 40 * (H_sample / W_sample)))
ax = plt.axes()
ax.set_facecolor('black')

my_cmap = cm1.magma

im = plt.scatter(nodes_interest_x * 1e3, nodes_interest_y * 1e3, s =100, c = I_VCF_global_final / np.max(I_VCF_global_final), cmap = my_cmap, zorder = 2)

cbar = fig.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=70)
cbar.ax.set_title('I', fontsize = 80, fontweight='bold', rotation=0, pad = 60)
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.scatter(x_trans * 1e3, y_trans * 1e3, s = 100, color = 'black', zorder = 3)

plt.tick_params(axis='both', which='major', labelsize=80)
plt.xlabel('Width (mm)', fontsize = 80, fontweight = 'bold')
plt.ylabel('Height (mm)', fontsize = 80, fontweight = 'bold')

ax.margins(0)
ax.set_aspect("equal", adjustable="box")

plt.savefig('ρ_cL', bbox_inches='tight', dpi=400)