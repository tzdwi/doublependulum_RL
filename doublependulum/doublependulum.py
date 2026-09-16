from jax import numpy as jnp
import numpy as np
import scipy
import scipy.integrate as itg

class DoublePendulum:
    """
    This is the object that'll get initialized and has methods to define its 
    state vector, and iterate it!
    """
    
    def __init__(self, 
                 mm : float = 1.0, 
                 m1 : float = 1.0, 
                 m2 : float = 1.0, 
                 l1 : float = 1.0, 
                 l2 : float = 1.0,
                 theta1_init : float | None = None,
                 theta2_init : float | None = None):
        self.mm = mm
        self.m1 = m1
        self.m2 = m2

        self.l1 = l1
        self.l2 = l2

        self.g = 9.81 # close enough!

        init_arr = np.zeros(6) # x_m, theta_1, theta_2, then time derivatives
        if theta1_init:
            init_arr[1] = theta1_init
        else:
            init_arr[1] = np.random.randn() #random

        if theta2_init:
            init_arr[2] = theta2_init
        else:
            init_arr[2] = np.random.randn()

        self.x = init_arr
        self.t = 0.0
            
        self._validate_state_arr()

        self.d11 = self.mm + self.m1 + self.m2
        self.d22 = (self.m1+self.m2)*self.l1**2.0
        self.d33 = self.m2*self.l2**2.0

        self.d12_part = (self.m1+self.m2)*self.l1
        self.d13_part = self.m2*self.l2
        self.d23_part = self.m2*self.l1*self.l2

        self.c12_part = -(self.m1+self.m2)*self.l1
        self.c13_part = -self.m2*self.l2
        self.c23_part = self.m2*self.l1*self.l2

        self.g2_part = self.g*(self.m1+self.m2)*self.l1
        self.g3_part = self.g*self.m2*self.l2

        self.H = np.array([1.0, 0, 0]).reshape(-1, 1)

        self.fac1 = np.zeros((6, 6))
        self.fac1[:3, 3:] = np.eye(3)

        self.fac2 = np.zeros((6, 1))
        self.fac3 = np.zeros((6, 1))

    def _validate_state_arr(self):
        if (-np.pi > self.x[1]) or (self.x[1] > np.pi):
            self.x[1] = np.arctan2(np.sin(self.x[1]),np.cos(self.x[1]))
        if (-np.pi > self.x[2]) or (self.x[2] > np.pi):
            self.x[2] = np.arctan2(np.sin(self.x[2]),np.cos(self.x[2]))

    def D(self):
        out = np.array([[self.d11, self.d12_part*np.cos(self.x[1]), self.d13_part*np.cos(self.x[2])],
                        [self.d12_part*np.cos(self.x[1]), self.d22, self.d23_part*np.cos(self.x[1]-self.x[2])],
                        [self.d13_part*np.cos(self.x[2]), self.d23_part*np.cos(self.x[1]-self.x[2]), self.d33]])
        return out

    def C(self):
        out = np.array([[0, self.c12_part*np.sin(self.x[1])*self.x[4], self.c13_part*np.sin(self.x[2])*self.x[5]],
                        [0, 0, self.c23_part*np.sin(self.x[1]-self.x[2])*self.x[5]],
                        [0, -self.c23_part*np.sin(self.x[1]-self.x[2])*self.x[4], 0]])
        return out

    def G(self):
        return np.array([0, self.g2_part*np.sin(self.x[1]), self.g3_part*np.sin(self.x[2])]).reshape(-1, 1)

    def compute_D_inv_vec_or_matrix(self, V):
        D = self.D()
        return scipy.linalg.solve(D, V)

    def _make_system(self):
        self.fac1[3:, 3:] = -self.compute_D_inv_vec_or_matrix(self.C())
        self.fac2[3:] = -self.compute_D_inv_vec_or_matrix(self.G())
        self.fac3[3:] = -self.compute_D_inv_vec_or_matrix(self.H)

    def _calc_max_timestep(self):
        """
        So the maximum timestep should depend on the amount of Kinetic Energy in the system and the applied force

        F*lengthscale
        KE
        """

    def step(self, F):
        self._make_system()
        def dynam_sys(t, x):
            return (self.fac1 @ x.reshape(-1, 1) + self.fac2 + self.fac3*F).flatten()
        solver = itg.RK45(dynam_sys, self.t, self.x, 5000, max_step=0.05)
        solver.step()
        self.x = solver.y
        self._validate_state_arr()
        self.t = solver.t

    def transform_cartesian(self, do_vels=False):
        x_m = self.x[0]
        x1 = x_m+self.l1*np.sin(self.x[1])
        x2 = x1 + self.l2*np.sin(self.x[2])

        y1 = -self.l1*np.cos(self.x[1])
        y2 = -self.l2*np.cos(self.x[2])

        out = [x_m, (x1, y1), (x2, y2), 0, (0,0), (0, 0)]

        if do_vels:
            v_m = self.x[3]
                
            vx1 = v_m + self.l1*np.cos(self.x[1])*self.x[4]
            vx2 = vx1 + self.l2*np.cos(self.x[2])*self.x[5]
    
            vy1 = self.l1*np.sin(self.x[1])*self.x[4]
            vy2 = vy1+self.l1*np.sin(self.x[2])*self.x[5]
    
            out[3] = v_m
            out[4] = (vx1, vy1)
            out[5] = (vx2, vy2)
        return out
        
        
        
            