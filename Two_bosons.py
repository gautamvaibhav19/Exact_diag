from qutip import *
import numpy as np
import scipy as sp
import os 
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import time


#### Misc. functions for use.   #################

def gaussian(x,mu,sig):
    return (1/np.sqrt(np.sqrt(2*np.pi)*sig**2))*np.exp(-(x - mu)**2/(4*sig**2))

def lewp_cost(H,x_hat,y_hat,px_hat,py_hat,mux,muy,qux,quy,psi,c=100):
    J = ((psi.dag() * H * psi)[0,0] + (c/2)*(psi.dag()*x_hat*psi - mux)[0,0]**2 + 
         (c/2)*(psi.dag()*(y_hat)*psi - muy)[0,0]**2 + (c/2)*(psi.dag()*px_hat*psi - qux)[0,0]**2+ 
         (c/2)*(psi.dag()*py_hat*psi - quy)[0,0]**2 + (c/2)*(psi.dag()*psi - 1)[0,0]**2)
    return J

def lewp_cost_prime(H,x_hat,y_hat,px_hat,py_hat,mux,muy,qux,quy,psi,c=100):
    J = (H * psi + c*(psi.dag()*x_hat*psi - mux)*x_hat*psi + c*(psi.dag()*(y_hat)*psi - muy)*y_hat*psi + 
         c*(psi.dag()*px_hat*psi - qux)*px_hat*psi + c*(psi.dag()*(py_hat)*psi - quy)*py_hat*psi +
         c*(psi.dag()*psi - 1)*psi)
    return J

#### Create position operator in the coordinate basis. ########
###### Input: Q = Number of qubits. , R = spatial trunctation. ############
###### Output: x_hat = position operator. ############# 


def create_xhat(Q,R):
    """
    Create position operator in the coordinate basis. 
    

    Parameters
    ----------
    Q : int
        Number of qubits.
    R : float
        Spatial truncation parameter.

    Returns
    -------
    x_hat : Qobj
        Position operator.

    """
    d = 2*R/(Q) 


    base_vecs = create_basis_vecs(Q)

    n_a = [(d/2)*(-(Q-1) + i*2) for i in range(Q)]
    #n_a = [((-((Q-1)*R)/Q) + i*d) for i in range(Q)]
    su = 0
    for i in range(Q):
        su = su + (n_a[i]*base_vecs[i]*base_vecs[i].dag())
    x_hat = su

    return x_hat


################################################################################
#### Create position operator in the FT basis. ########
###### Input: Q = Number of qubits. , R = spatial trunctation. ############
###### Output: x_hat = position operator. ############# 


def createFT_xhat(Q,R):
    """
    Create position operator in the momentum basis. 
    

    Parameters
    ----------
    Q : int
        Number of qubits.
    R : float
        Spatial truncation parameter.

    Returns
    -------
    ft_xhat : Qobj
        Position operator.

    """

    x_hat = create_xhat(Q, R)
    
    ft_base = FT_basis_vecs(Q)
    ft_xhat = np.empty([Q,Q], dtype = complex)
    for i in range(Q):
        for j in range(Q):
            ft_xhat[i,j] = (ft_base[j].dag()*x_hat*ft_base[i])[0,0]
    ft_xhat = Qobj(ft_xhat)
    
    ft_xsq = np.empty([Q,Q], dtype = complex)
    for i in range(Q):
        for j in range(Q):
            ft_xsq[i,j] = (ft_base[j].dag()*(x_hat**2)*ft_base[i])[0,0]
    ft_xsq = Qobj(ft_xhat)
    return ft_xhat, ft_xsq

################################################################################

###### Create set of basis vectors. ############
###### Input: Q = Number of qubits.  ############
###### Output: List of basis vectors |00...0>, |00...01> etc. #############

def create_basis_vecs(Q):
    """
    Create set of basis vectors for given lambda.

    Parameters
    ----------
    Q : int
        Size of Hamiltonian truncation level.

    Returns
    -------
    base_vec_list : list of Qobj
        List of basis vectors |00...0>, |00...01> etc.

    """
    
    if (Q % 2 == 0):
        base_vec_list = [basis(Q,i) for i in range(Q)]
    else:
        raise Exception("Truncation level must be even.")
    return base_vec_list



################################################################################

###### Create set of basis vectors. ############
###### Input: Q = Number of qubits.  ############
###### Output: List of basis vectors |00...0>, |00...01> etc. #############

def FT_basis_vecs(Q):
    """
    Create set of basis vectors in the momentum basis.

    Parameters
    ----------
    Q : int
        Size of Hamiltonian truncation level.

    Returns
    -------
    base_vec_list : list of Qobj
        List of basis vectors |00...0>, |00...01> etc.

    """
    cord_basis = create_basis_vecs(Q)
    n_til = [i for i in range(int(-Q/2), int(Q/2))]
    #n_til = [(-(Q-1) + i*2) for i in range(Q)]
    n_a = [(-(Q-1) + i*2) for i in range(Q)]
    base_vec_list = []
    for til in n_til:
        s = 0
        for i in range(len(n_a)):
            s = s + (1/np.sqrt(Q))*np.exp((2j * np.pi / Q)*(til+0.5)*(n_a[i]/2))*cord_basis[i]
        base_vec_list.append(s)
    
    return base_vec_list


################################################################################

def create_phat(Q,R):
    """
    Create momentum operator in the momentum basis. 
    

    Parameters
    ----------
    Q : int
        Number of qubits.
    R : float
        Spatial truncation parameter.

    Returns
    -------
    x_hat : Qobj
        Position operator.

    """
    base = FT_basis_vecs(Q)
    n_til = [(np.pi/R)*(i+0.5) for i in range(int(-Q/2), int(Q/2))]
    su = 0
    for i in range(Q):
        su = su + (n_til[i]*base[i]*base[i].dag())
    p_hat = su

    return p_hat

################################################################################

def TwoBoson_ham(Q,R,coup= 1):
    """
    Creates truncated Hamiltonian for the two boson model for given number of qubits and truncation.
    Uses momentum basis.
    Parameters
    ----------
    Q : int
        Truncation level.
    R : float
        Spatial truncation parameter.
    coup : float, optional
        Coupling. The default is 1.


    Returns
    -------
    Ham : csc_matrix
        Hamiltonian of the two boson model.
    list
        List of position operators for x and y resp.
    list
        List of momentum operators for x and y resp..

    """
   
    pos = create_xhat(Q, R)
    Id = identity(Q)
    
    x_hat = tensor(pos,Id)
    y_hat = tensor(Id,pos)
    
    mom = create_phat(Q, R)
    px_hat = tensor(mom,Id)
    py_hat = tensor(Id,mom)
    
    Ham = px_hat**2/2 + py_hat**2/2 + coup*(x_hat**2 * y_hat**2)/4

    return Ham,[x_hat,y_hat],[px_hat,py_hat]

#################################################################################

def OptimumR(Q_max, R_max,Q_min=4,R_min=1,coup= 1):
    """
    
    Plots the ground state energy vs R curve for Q = [Q_min,Q_max] and returns dataframe of optimum values for Two Boson model.
    Parameters
    ----------
    Q_max : int
        Max number of qubits.
    R_max : int
        Max value of spatial truncation.
    Q_min : int, optional
        Min number of qubits. The default is 4.
    R_min : int, optional
        Min value of spatial truncation. The default is 1.
    coup : float, optional
        Coupling. The default is 1.

    Returns
    -------
    df : Dataframe
        Data frame showing the value of R for which deviation from groundstate is minimum.

    """
    data = []
    x_list = np.arange(R_min,R_max,0.1)
    
    fig,ax = plt.subplots()
    for q in range(Q_min,Q_max+1,2):
        
        gs_enelist = np.array([(abs(TwoBoson_ham(q, r)[0].eigenenergies(sparse = True, tol= 1e-06, eigvals = 1)[0])) for r in x_list])
        dat = [q,x_list[np.argmin(gs_enelist)]]
        data.append(dat)
        
        ax.plot(x_list,gs_enelist, label = str(q))
        #ax.set_yscale('log')
        ax.legend(loc='upper right', title = r"Q")
        ax.set_xlabel(r"$R$", fontsize=18)
        ax.set_ylabel(r"$|E_{gs}|$", fontsize=18)
        fig.suptitle("Two Boson")
        fig.savefig("C:\\Users\\gauta\\OneDrive\\Desktop\\Codes\\Exact_Diag\\Two_bos_plots\\TwoBos_gs_alt.pdf",bbox_inches= 'tight',dpi=300 )
    
    plt.show()
    
    data = np.array(data)
    cols = ["Q","R_min_gs"]
    
    df = pd.DataFrame(data, columns = cols)
    return df

##################################################################################
def LowEnergy_wp(Q,R,mu_x,mu_y,qu_x,qu_y,lr = 1e-06,tole = 0.05, c = 100, beta=0.01, coup= 1):
    """
    Create a low energy wavepacket for given Q and R centered around mu in position space and qu in momentum space.
    
    Parameters
    ----------
    Q : int
        Number of qubits.
    R : float
        Spatial truncation parameter.
    mu_x : float
         Center of wp in position space along x-axis.
    mu_y : float
         Center of wp in position space along y-axis.         
    qu_x : float
         Center of wp in momentum space along px-axis.
    qu_y : float
         Center of wp in momentum space along py-axis.     
    lr : float, optional
        Learning rate for gradient descent. The default is 0.0001.
    tole : float, optional
        Learning rate for gradient descent. The default is 0.05.
    c : float, optional
        Weight parameter in the cost function.. The default is 100.
    beta : float, optional
        Momentum coefficient. The default is 0.01.
    coup : float, optional
        Coupling. The default is 1.

    Returns
    -------
   psi_f : Qobj
       |psi> = psi_i |n_i> where |n_i> are bases states.

    """
    
    H,pos_list,p_list = TwoBoson_ham(Q, R, coup= coup)
    x_hat, y_hat = pos_list
    px_hat, py_hat = p_list
    
    it = 0
    
    d = 2*R/(Q)    
    
    x_list = [(d/2)*(-(Q-1) + i*2) for i in range(Q)]
    
    base_vecs = create_basis_vecs(Q)
    
    psix_i=0
    gx_list = np.array([gaussian(x, mu_x, 1) for x in x_list]) 
    for i in range(len(gx_list)): 
        psix_i = psix_i +  gx_list[i]*base_vecs[i]
        
    psiy_i=0
    gy_list = np.array([gaussian(x, mu_y, 1) for x in x_list]) 
    for i in range(len(gy_list)): 
        psiy_i = psiy_i +  gy_list[i]*base_vecs[i]
    
    psi_i = tensor(psix_i,psiy_i)
    
    J_i = 10
    J_f = 0
    v_i = 0
    while abs(J_i-J_f) > tole:
        J_i = abs(lewp_cost(H, x_hat, y_hat, px_hat, py_hat, mu_x,mu_y,qu_x,qu_y, psi_i,c = c))
        v_f = beta*v_i + (1-beta)*lewp_cost_prime(H, x_hat, y_hat, px_hat, py_hat, mu_x, mu_y, qu_x, qu_y, psi_i,c = c)
        psi_f = psi_i - lr*v_f
        it = it+1
        
        J_f = abs(lewp_cost(H, x_hat, y_hat, px_hat, py_hat, mu_x,mu_y,qu_x,qu_y, psi_f,c = c))
        v_i = v_f
        print(J_f)
        psi_i = psi_f
        
    print(it)
        
    return psi_f


##################################################################################

def evolve_state(Q,Ham, state, t):
    """
    Evolve a given state or with the given hamiltonian.

    Parameters
    ----------
    Q : int
        Number of qubits.
    Ham : Qobj
        Hamiltonian of the system under study.
    state : Qobj
        The state or wavepacket to be evolved.
    t : float
        Time t to evolve to.
    Returns
    -------
    ev_state : Qobj
        |psi(t)> = exp(-iHt)|psi(0)>.

    """
    h_p = sp.sparse.csc_matrix(Ham.data)
    expiH = Qobj(sp.sparse.linalg.expm(-1j * h_p * t), dims = Ham.dims, shape = Ham.shape)
    ev_state = expiH * state
    
    return ev_state

##################################################################################

def expec_xhat(Q,R,H,x_hat,y_hat,state, t):
    """
    Compute the expectation value of x and variance for a state evolved to time t.    
    
    Parameters
    ----------
    Q : int
        Number of qubits.
    R : float
        Spatial truncation parameter.
    H : Qobj
        Hamiltonian of the system under study.
    x_hat : Qobj
        Position operator along x.
    y_hat : Qobj
        Position operator along y.
    state : Qobj
        The state or wavepacket to be evolved.
    t : float
        Time t to evolve to.

    Returns
    -------
    complex
        <psi(t)|x_hat|psi(t)>.
    complex
        <psi(t)|y_hat|psi(t)>.

    """
    
    ev_state = evolve_state(Q, H, state, t)
    
    r = ev_state.dag()*(x_hat)*ev_state
    r1 = ev_state.dag()* (y_hat) *ev_state
    r_sig = (ev_state.dag()* (x_hat)**2 *ev_state) - r[0,0]**2
    r1_sig = (ev_state.dag()* (y_hat)**2 *ev_state) - r1[0,0]**2
    return r[0,0],r1[0,0],r_sig[0,0],r1_sig[0,0]

#################################################################################

def plot_time_ev(Q,R,state,ini_cond,t_max = 10, n_points=50, coup= 1):
    """
    Plot the expectation value of x and variance for a state evolved to time t.

    Parameters
    ----------
    Q : int
        Number of qubits.
    R : float
        Spatial truncation parameter.
    state : Qobj
        The state or wavepacket to be evolved.
    ini_cond : list
        Initial data information.
    t_max : float, optional
        Maximum time for evolution. The default is 10.
    n_points : int, optional
        Number of times between 0 and t_max. The default is 50.
    coup : float, optional
        Coupling. The default is 1.

    Returns
    -------
    int
        Plots the curve.

    """
    H,pos_list,p_list = TwoBoson_ham(Q, R, coup= coup)
    x_hat, y_hat = pos_list
    px_hat, py_hat = p_list
    x,y,px,py = ini_cond    
    sup_ini = "("+str(x)+","+str(y)+","+str(px)+","+str(py)+")"
    
    times = np.linspace(0,t_max,n_points)
    result = [expec_xhat(Q, R, H, x_hat,y_hat, state, t) for t in times]
    result = np.array(result)
    fig, ax = plt.subplots() 
    ax.plot(times, result[:,0], label = r"$\langle x \rangle$")
    ax.plot(times, result[:,1], label = r"$\langle y \rangle$")
    ax.plot(times, result[:,2], label = r"$\langle (x - \langle x \rangle )^2 \rangle$")
    ax.plot(times, result[:,3], label = r"$\langle (y - \langle y \rangle )^2 \rangle$")
    #ax.plot(result.times, result.expect[1]) 
    ax.set_xlabel('Time') 
    ax.set_ylabel('Expectation values')
    ax.legend(loc='upper right')
    fig.suptitle("Q= "+str(Q)+ ", R= "+str(R)+ ", Coupling= "+str(coup)+", ini="+sup_ini)
    fig.savefig("C:\\Users\\gauta\\OneDrive\\Desktop\\Codes\\Exact_Diag\\Q"+str(Q)+"_R"+str(R).translate({ord(c): None for c in '.'})+"_c"+str(coup).translate({ord(c): None for c in '.'})+"_x"+str(ini_cond[0])+"_y"+str(ini_cond[1]).translate({ord(c): None for c in '.'})+".pdf",bbox_inches= 'tight',dpi=300 )
    plt.show()

    return 0    


##################################################################################
Q=4
R=8
cp_list = [1]
y_list = [1]

OptimumR(16, 20)
H.eigenenergies(sparse = True, tol= 1e-06, eigvals = 1)

for y in y_list:
    for cp in cp_list:
        x=2
        px=0
        py=0
        H,pos_list,p_list = TwoBoson_ham(Q, R,coup = cp)
        x_hat, y_hat = pos_list
        px_hat, py_hat = p_list
        ini_cond = [x,y,px,py]
        lewp = LowEnergy_wp(Q, R, x, y, px, py,tole = 1e-06,c = 10000, coup= cp)
        plot_time_ev(Q, R, lewp,ini_cond,t_max = 300,n_points=300,coup= cp)
        
plot_time_ev(Q, R, lewp,ini_cond,t_max = 300,n_points=300,coup= cp)


about()


