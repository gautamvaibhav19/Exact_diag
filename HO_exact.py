from qutip import *
import numpy as np
import scipy as sp
import os 
import itertools
import matplotlib.pyplot as plt
import pandas as pd

################################################################################

#### Misc. functions for use.   #################

def Gaussian_cost(x_hat,psi,c,mu,sig):
    J = (1/2)*(psi.dag()*x_hat*psi - mu)[0,0]**2 + (1/2)*(psi.dag()*(x_hat**2)*psi - mu**2 - sig**2)[0,0]**2 + (c/2)*(psi.dag()*psi - 1)[0,0]**2
    return J

def Gaus_cost_prime(x_hat,psi,c,mu,sig):
     J = (psi.dag()*x_hat*psi - mu)*x_hat*psi + (psi.dag()*(x_hat**2)*psi - mu**2 - sig**2)*(x_hat**2)*psi + c*(psi.dag()*psi - 1)*psi
     return J
 
def gaussian(x,mu,sig):
    return np.sqrt(np.sqrt(2*np.pi)*sig**2)*np.exp(-(x - mu)**2/(4*sig**2))

def lewp(H,x_hat,p_hat,mu,qu,psi,c=100):
    J = psi.dag() * H * psi + (c/2)*(psi.dag()*x_hat*psi - mu)[0,0]**2 + (c/2)*(psi.dag()*(p_hat)*psi - qu)[0,0]**2 + (c/2)*(psi.dag()*psi - 1)[0,0]**2
    return J
################################################################################
 
#### Create position operator in the coordinate basis. ########
###### Input: Q = Number of qubits. , R = spatial trunctation. ############
###### Output: x_hat = position operator. ############# 


def create_xhat(Q,R):
    d = 2*R/(2**Q) 


    base_vecs = create_basis_vecs(Q)

    I = identity(2)
    Z = sigmaz()

    prod_list = [I]*Q
    x_hat = 0
    for i in range(Q):
        op_list = prod_list.copy()
        op_list[i] = 2**(Q-1-i) * Z
        x_hat = x_hat + tensor(op_list)
        
    x_hat = -d * x_hat / 2

    return x_hat

################################################################################

###### Create set of basis vectors. ############
###### Input: Q = Number of qubits.  ############
###### Output: List of basis vectors |00...0>, |00...01> etc. #############

def create_basis_vecs(Q):

    H_sp = [2]*Q
    
    bas = []
    for numbers in itertools.product([0, 1], repeat=Q):
        bas.append(list(numbers))
    
    base_vec_list = []
    for exc in bas:
        basis_vec = basis(H_sp,exc)
        base_vec_list.append(basis_vec)
    
    return base_vec_list

###############################################################################

##### Creates Hamiltonian for the harmonic oscillator.   #############
###### Input: Q = Number of qubits.     R = spatial truncation   ############
###### Output: Hamiltonian as a Qobj.  #############


def create_Hamiltonian_HO(Q,R):
    d = 2*R/(2**Q) 

    base_vecs = create_basis_vecs(Q)
    
    I = identity(2)
    Z = sigmaz()
    
    prod_list = [I]*Q
    x_hat = 0
    for i in range(Q):
        op_list = prod_list.copy()
        op_list[i] = 2**(Q-1-i) * Z
        x_hat = x_hat + tensor(op_list)
        
    x_hat = -d * x_hat / 2
        
    Id = tensor(prod_list)
    
    S = 0
    for i in range(len(base_vecs)-1):
        S = S + base_vecs[i]*base_vecs[i+1].dag() + base_vecs[i+1]*base_vecs[i].dag()
    
    p2_hat = (2*Id - S)/(d**2) 
    
    p_hat = sqrtm(p2_hat)
    
    Ham_ho = p2_hat/2 +  x_hat**2/2 
    
    return Ham_ho,x_hat,p_hat




###############################################################################

##### Creates Hamiltonian for the anharmonic oscillator.   #############
###### Input: Q = Number of qubits.     R = spatial truncation   ############
###### Output: Hamiltonian as a Qobj.  #############




def create_Hamiltonian_AnHO(Q,R):
    d = 2*R/(2**Q) 

    base_vecs = create_basis_vecs(Q)
    
    I = identity(2)
    Z = sigmaz()
    
    prod_list = [I]*Q
    x_hat = 0
    for i in range(Q):
        op_list = prod_list.copy()
        op_list[i] = 2**(Q-1-i) * Z
        x_hat = x_hat + tensor(op_list)
        
    x_hat = -d * x_hat / 2
        
    Id = tensor(prod_list)
    
    S = 0
    for i in range(len(base_vecs)-1):
        S = S + base_vecs[i]*base_vecs[i+1].dag() + base_vecs[i+1]*base_vecs[i].dag()
    
    p2_hat = (2*Id - S)/(d**2) 
    
    p_hat = sqrtm(p2_hat)
    
    Ham_anho = p2_hat/2 +  x_hat**4/4 
    
    return Ham_anho,x_hat,p_hat

###############################################################################

##### Plots the ground state energy vs R curve for various Q and returns df of optimum values.  #############
###### Input: Q_max = Max. number of qubits.     R_max = Max. value of spatial truncation   ############
###### Output: Dataframe giving R at which ground state is min.  #############


def OptimumR_HO(Q_max, R_max,Q_min=2,R_min=1):
    data = []
    x_list = np.arange(R_min,R_max,0.1)
    
    fig,ax = plt.subplots()
    for q in range(Q_min,Q_max+1):
        gs_enelist = np.array([(abs(create_Hamiltonian_HO(q, r).eigenenergies(sparse = True, tol= 1e-06, eigvals = 1)[0] - 0.5)) for r in x_list])
        dat = [q,x_list[np.argmin(gs_enelist)]]
        data.append(dat)
        
        ax.plot(x_list,gs_enelist, label = str(q))
        ax.set_yscale('log')
        ax.legend(loc='upper right', title = r"Q")
        ax.set_xlabel(r"$R$", fontsize=18)
        ax.set_ylabel(r"$|E - 0.5|$", fontsize=18)
        fig.suptitle("Harmonic Osc.")
        fig.savefig("C:\\Users\\gauta\\OneDrive\\Desktop\\Codes\\Exact_Diag\\HO_gs.pdf",bbox_inches= 'tight',dpi=300 )
    
    plt.show()
    
    data = np.array(data)
    cols = ["Q","R_min_gs"]
    
    df = pd.DataFrame(data, columns = cols)
    return df


#################################################################################

##### Plots the ground state energy vs R curve for various Q and returns df of optimum values for AHO.  #############
###### Input: Q_max = Max. number of qubits.     R_max = Max. value of spatial truncation   ############
###### Output: Dataframe giving R at which ground state and 1st exc state resp is min.  #############



def OptimumR_AnHO(Q_max, R_max):
    data = []
    x_list = np.arange(1,R_max,0.1)
    
    fig,ax = plt.subplots()
    fig1,ax1 = plt.subplots()
    for q in range(2,Q_max+1):
        gs_enelist = np.array([(abs(create_Hamiltonian_AnHO(q, r).eigenenergies(sparse = True, tol= 1e-06, eigvals = 2)[0] - 0.420804974)) for r in x_list])
        exc_enelist = np.array([(abs(create_Hamiltonian_AnHO(q, r).eigenenergies(sparse = True, tol= 1e-06, eigvals = 2)[1] - 1.5079012411)) for r in x_list])
        dat = [q,x_list[np.argmin(gs_enelist)],x_list[np.argmin(exc_enelist)]]
        data.append(dat)
        
        ax.plot(x_list,gs_enelist, label = str(q))
        ax.set_yscale('log')
        ax.legend(loc='upper right', title = r"Q")
        ax.set_xlabel(r"$R$", fontsize=18)
        ax.set_ylabel(r"$|E_0 - 0.420804974|$", fontsize=18)
        fig.suptitle("Anharmonic Osc. ground state")
        fig.savefig("C:\\Users\\gauta\\OneDrive\\Desktop\\Codes\\Exact_Diag\\AnHO_gs.pdf",bbox_inches= 'tight',dpi=300 )
        
        ax1.plot(x_list,exc_enelist, label = str(q))
        ax1.set_yscale('log')
        ax1.legend(loc='upper right', title = r"Q")
        ax1.set_xlabel(r"$R$", fontsize=18)
        ax1.set_ylabel(r"$|E_1 - 1.507901|$", fontsize=18)
        fig1.suptitle("Anharmonic Osc. excited state")
        fig1.savefig("C:\\Users\\gauta\\OneDrive\\Desktop\\Codes\\Exact_Diag\\AnHO_exc.pdf",bbox_inches= 'tight',dpi=300 )
    
    plt.show()
    
    data = np.array(data)
    cols = ["Q","R_min_gs","R_min_exc"]
    
    df = pd.DataFrame(data, columns = cols)
    
    return df
################################################################################            

######### Create a gaussian wavepacket for given Q and R with mean mu and variance sig. ###########
###### Input: Q = qubits. R = spatial truncation, c = norm weight, mu = center of wp, sig = width of wp   ############
######          lr = learning rate, tole = tolerance to cost  ###########################
###### Output: |psi> = psi_i |n_i>  (wavepacket) #############


def Gaussian_wp(Q,R,c,mu,sig,lr,tole): 
    it = 0
        
    x_hat = create_xhat(Q, R)
    
    x_list = [((-(2**Q-1)/(2**Q))*R + d*n) for n in range(2**Q)]
    
    psi_i=0
    g_list = np.array([gaussian(x, 0, 1) for x in x_list]) 
    for i in range(len(g_list)): 
        psi_i = psi_i +  g_list[i]*base_vecs[i]
    
    
    J = abs(Gaussian_cost(x_hat, psi_i, c, mu, sig))
    while J > tole:
        psi_f = psi_i - lr*Gaus_cost_prime(x_hat, psi_i, c, mu, sig)
        it = it+1
        J = abs(Gaussian_cost(x_hat, psi_f, c, mu, sig)) 
        psi_i = psi_f
        
    print(it)
        
    return psi_f


################################################################################

###### Plotting <x> and < (x- <x>)^2 > ############

################################################################################

######### Evolve a given state or with the given hamiltonian. ###########
###### Input: Q = qubits. Ham = hamiltonian, state = state to　evovlve, t = time to evolve to   ############
###### Output: |psi(t)> = exp(-iHt)|psi(0)>  (wavepacket) #############
 
def evolve_state(Q,Ham, state, t):
    h_p = sp.sparse.csc_matrix(Ham.data)
    expiH = Qobj(sp.sparse.linalg.expm(-1j * h_p * t), dims = [list([2]*Q), list([2]*Q)], shape = (2**Q, 2**Q))
    ev_state = expiH * state
    
    return ev_state


################################################################################

######### Compute the expec value of x and variance for a state evolved to time t. ###########
###### Input: Q = qubits. Ham = hamiltonian, state = state to　evovlve, t = time to evolve to   ############
######      x_hat = position operator    ###########################
###### Output: <psi(t)|x_hat|psi(t)>  and <psi(t)|(x_hat- <x_hat>)^2|psi(t)> #############




def expec_xhat(Q,R,H,x_hat,state, t):
    
    ev_state = evolve_state(Q, H, state, t)
    
    r = ev_state.dag()*(x_hat)*ev_state
    r1 = ev_state.dag()* (x_hat-r[0,0])**2 *ev_state
    return r[0,0],r1[0,0]


###################################################################################
######### Plot the expec value of x and variance for a state evolved to time t. ###########
###### Input: Q = qubits. Ham = hamiltonian, state = state to　evovlve, t = time to evolve to   ############
######      x_hat = position operator, R = spatial trunc.    ###########################
###### Output: <psi(t)|x_hat|psi(t)>  and <psi(t)|(x_hat- <x_hat>)^2|psi(t)> #############



def plot_time_ev(Q,R,state,model = "HO",t_max = 10, n_points=50):
    if model == "HO":
        H,x_hat,p_hat = create_Hamiltonian_HO(Q, R)
    elif model == "AnHO":
        H,x_hat,p_hat = create_Hamiltonian_AnHO(Q, R)
    else:
        print("Model undefined.")
    
    times = np.linspace(0,t_max,n_points)
    result = [expec_xhat(Q, R, H, x_hat, state, t) for t in times]
    result = np.array(result)
    fig, ax = plt.subplots() 
    ax.plot(times, result[:,0], label = r"$\langle x \rangle$")
    ax.plot(times, result[:,1], label = r"$\langle (x - \langle x \rangle )^2 \rangle$") 
    #ax.plot(result.times, result.expect[1]) 
    ax.set_xlabel('Time') 
    ax.set_ylabel('Expectation values')
    ax.legend(loc='upper right')
    fig.suptitle(model + ", Q=" + str(Q) + ", R=" + str(R))
    fig.savefig("C:\\Users\\gauta\\OneDrive\\Desktop\\Codes\\Exact_Diag\\HO_Q"+str(Q)+"_R"+str(R)+".pdf",bbox_inches= 'tight',dpi=300 )
    plt.show()

    return 0    


################################################################################





Q = [4]
R = [5]
mu = 1
c = 100
sig = 1/2 
tole = 0.0005
lr = 0.0001


df_ho = OptimumR_HO(8, 10)
df_anho = OptimumR_AnHO(8,10)

for q in Q:
    for r in R:
        g_wp = Gaussian_wp(q, r,c , mu, sig, lr, tole)
        plot_time_ev(q, r, g_wp,t_max = 10)

x_hat = create_xhat(4, 5)

g_wp.dag()* x_hat * g_wp
g_wp.dag()* x_hat**2 * g_wp 
Gaussian_cost(x_hat, g_wp, 100, mu, sig)