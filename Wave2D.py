import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        # self.xji, self.yij = ...
        self.N = N
        self.h = 1/N
        x = np.linspace(0, 1, N+1)
        self.x = x
        self.xij, self.yij = np.meshgrid(x, x, indexing='ij', sparse=sparse)
        return self.xij, self.yij

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        D /= self.h**2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return np.sqrt(self.c**2 * (self.mx**2 + self.my**2))*np.pi

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        xij, yij = self.xij, self.yij
        dt = self.dt
        D2 = self.D2(N)
        #Use exact solution to initialize first two time steps
        U0 = sp.lambdify((x, y, t), self.ue(mx, my))(xij, yij, 0.0)
        U1 = U0 + (self.c*self.dt)**2 / 2 * (D2@U0 + U0@D2.T)
        return U0, U1

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl*self.h/self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        xij, yij = self.xij, self.yij
        uj = sp.lambdify((x, y, t), self.ue(self.mx, self.my))(xij, yij, t0)
        h = self.h
        return np.sqrt(h**2*np.sum((uj-u)**2))

    def apply_bcs(self, u):
        u[0] = 0
        u[-1] = 0
        u[:,0] = 0
        u[:,-1] = 0
        return u

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        xij, yij = self.create_mesh(N)
        self.Nt = Nt
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my
        
        dt = self.dt
        D2 = self.D2(N)
        #Init storage dict
        self.data = {}
        #Initial values
        Um2, Um1 = self.initialize(N, mx, my)
        if store_data == 1:
            self.data[dt] = Um1
            
        U = np.zeros((N+1, N+1))
        for i in range(2,Nt+1):
            U = c**2*dt**2 * (D2 @ Um1 + Um1 @ D2.T) + 2*Um1 - Um2
            self.apply_bcs(U)
            
            if store_data != -1 and i % store_data == 0:
                self.data[i*dt] = U.copy()
            
            #switch variables
            Um2 = Um1
            Um1 = U
        
        if store_data == -1:
            l2_err = self.l2_error(U, Nt*dt)
            return (self.h, l2_err)
        else:
            return self.data

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            h.append(dx)
            E.append(err)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, 1] = 2
        D[-1, -2] = 2
        D /= self.h**2
        return D

    def ue(self, mx, my):
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self, u):
        pass

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(m=8, mx=2, my=2, Nt= 5, cfl=1/np.sqrt(2))
    assert abs(E[-1]) < 1e-12
    
    sol = Wave2D_Neumann()
    r, E, h = sol.convergence_rates(m=8, mx=2, my=2, Nt = 5, cfl=1/np.sqrt(2))
    assert abs(E[-1]) < 1e-12
    

