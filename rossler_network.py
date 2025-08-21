import torch
from gpytorch import utils
import numpy as np
from scipy.integrate import solve_ivp

def hermitian_matrix(N):
    np.random.seed(100)
    c = torch.tensor([0, *(np.random.rand(N-1))], dtype=torch.float32)
    sym_mat= utils.toeplitz.sym_toeplitz(c)
    d = torch.tensor([0, *(np.random.rand(N-1))], dtype=torch.float32)
    res_skew = utils.toeplitz.sym_toeplitz(d)
    skew_mat=torch.tril(res_skew) - torch.triu(res_skew)
    hermitian_mat=sym_mat + skew_mat*1j
    return np.array(hermitian_mat)



def ross_preprocess(x, mode, t_end):
    sr=50000
    mean_sr = 500
    if mode == 'training':
        x_reshaped = x[:,2*sr:(2+t_end)*sr].reshape(-1,mean_sr*t_end,100)
    elif mode == 'testing':
        x_reshaped = x[:,(2+t_end)*sr:(2+2*t_end)*sr].reshape(-1,mean_sr*t_end,100)
    elif mode == 'whole':
        x_reshaped = x[:,2*sr:(2+2*t_end)*sr].reshape(-1,2*mean_sr*t_end,100)

    x_mean = np.mean(x_reshaped, axis = 2)
    x_data = torch.tensor(x_mean.T, dtype = torch.float)
    x_data = (x_data - x_data.min())/(x_data.max()-x_data.min())
    x_data = x_data.unsqueeze(dim=0)
    return x_data



class RosslerSystem:
    def __init__(self, N, complex_wts, omega,const, a, b, c, k, Iext, d, t_span, num_points):
        self.N = N
        self.a = a
        self.b = b
        self.c = c
        self.k = k
        self.Iext = Iext
        self.d = d
        self.t_span = t_span
        self.num_points = num_points
        self.const = const
        self.complex_wts = complex_wts
        self.omega = omega

    def initialize_values(self,N):
      x_initial = np.random.rand(N)
      x_initial[0]=0.1
      y_initial = np.random.rand(N)
      y_initial[0]=0.1
      z_initial = np.random.rand(N)
      z_initial[0]=0.1
      initial_vals = np.concatenate([x_initial, y_initial, z_initial])
      return initial_vals

    def _rossler(self, t, xyz):
        N, a, b, c, k, Iext, d, omega, const = self.N, self.a, self.b, self.c, self.k, self.Iext, self.d, self.omega, self.const
        x, y, z = xyz[:N], xyz[N:2*N], xyz[2*N:3*N]
        x_mean = np.mean(x)

        # complex_coupling
        complex_num = x + 1j * y
        complex_pdt = np.matmul(self.complex_wts, complex_num)

        dxdt = -omega * y - z + k * (d - x_mean) + const * complex_pdt.real
        dydt = omega * x + a * y + const * complex_pdt.imag
        dzdt = b + z * (x - c) + Iext
        return np.concatenate([dxdt, dydt, dzdt])

    def solve(self):
        initial_vals = self.initialize_values(self.N)
        t_eval = np.linspace(*self.t_span, self.num_points)
        sol = solve_ivp(
            self._rossler,
            self.t_span,
            initial_vals,
            t_eval=t_eval,
            vectorized=True,
            method='LSODA',
            dense_output=True
        )
        x = sol.y[:self.N]
        y = sol.y[self.N:2*self.N]
        z = sol.y[2*self.N:]
        return x, y, z
    

