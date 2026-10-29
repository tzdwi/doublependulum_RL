import numpy as np
from numba import njit


@njit(cache=True)
def _solve_symmetric_3x3(a, b, c, d, e, f, r0, r1, r2):
    """Solve [[a,b,c], [b,d,e], [c,e,f]] * x = rhs."""
    c00 = d * f - e * e
    c01 = c * e - b * f
    c02 = b * e - c * d
    c11 = a * f - c * c
    c12 = b * c - a * e
    c22 = a * d - b * b
    determinant = a * c00 + b * c01 + c * c02
    return (
        (c00 * r0 + c01 * r1 + c02 * r2) / determinant,
        (c01 * r0 + c11 * r1 + c12 * r2) / determinant,
        (c02 * r0 + c12 * r1 + c22 * r2) / determinant,
    )


@njit(cache=True)
def _derivative(x, force, params):
    d11, d22, d33, d12_part, d13_part, d23_part, c12_part, c13_part, c23_part, g2_part, g3_part = params
    theta1, theta2 = x[1], x[2]
    theta1_dot, theta2_dot = x[4], x[5]
    cos1, cos2 = np.cos(theta1), np.cos(theta2)
    sin1, sin2 = np.sin(theta1), np.sin(theta2)
    cos12, sin12 = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    cart_accel, theta1_accel, theta2_accel = _solve_symmetric_3x3(
        d11,
        d12_part * cos1,
        d13_part * cos2,
        d22,
        d23_part * cos12,
        d33,
        force - c12_part * sin1 * theta1_dot**2 - c13_part * sin2 * theta2_dot**2,
        -c23_part * sin12 * theta2_dot**2 - g2_part * sin1,
        c23_part * sin12 * theta1_dot**2 - g3_part * sin2,
    )

    return np.array((
        x[3], theta1_dot, theta2_dot, cart_accel, theta1_accel, theta2_accel,
    ))


@njit(cache=True)
def _rk4_step(x, force, dt, params):
    k1 = _derivative(x, force, params)
    k2 = _derivative(x + 0.5 * dt * k1, force, params)
    k3 = _derivative(x + 0.5 * dt * k2, force, params)
    k4 = _derivative(x + dt * k3, force, params)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

class DoublePendulum:
    """
    This is the object that'll get initialized and has methods to define its 
    state vector, and iterate it!
    """
    
    def __init__(self, 
                 mm : float = 5.0, 
                 m1 : float = 0.5, 
                 m2 : float = 0.5, 
                 l1 : float = 1.0, 
                 l2 : float = 1.0,
                 theta1_init : float | None = None,
                 theta2_init : float | None = None,
                 dt : float = 0.03):
        
        super(DoublePendulum, self).__init__()
        
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
        self.dt = dt
            
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

        self._params = np.array((
            self.d11, self.d22, self.d33,
            self.d12_part, self.d13_part, self.d23_part,
            self.c12_part, self.c13_part, self.c23_part,
            self.g2_part, self.g3_part,
        ))

    def _initialize(self, seed):
        rng = np.random.default_rng(seed)
        init_arr = np.zeros(6) # x_m, theta_1, theta_2, then time derivatives
        init_arr[1] = np.pi+0.01*rng.uniform(-1, 1) #random
        init_arr[2] = np.pi+0.01*rng.uniform(-1, 1)
        init_arr[4] = 0.0001*rng.normal()
        init_arr[5] = 0.0001*rng.normal()
        self.x = init_arr
        self._validate_state_arr()
        self.t = 0

    def _validate_state_arr(self):
        """
        If down is theta=0, and our goal is up (theta=pi), wrap into
        [-pi/2, 3pi/2]. Non-finite angles are left unchanged.
        """
        low = -np.pi / 2
        high = 3 * np.pi / 2
        for i in (1, 2):
            theta = self.x[i]
            if np.isfinite(theta) and (theta < low or theta > high):
                self.x[i] = low + np.mod(theta - low, 2 * np.pi)
    
    def bounce(self, rail_pos=50.0):
        self.x[3] = -self.x[3]
        self.x[0] = np.clip(self.x[0], -rail_pos, rail_pos)

    def step(self, F):
        self.x = _rk4_step(self.x, float(np.asarray(F).item()), self.dt, self._params)
        self._validate_state_arr()
        self.t += self.dt

    def transform_cartesian(self, do_vels=False):
        x_m = self.x[0]
        x1 = x_m+self.l1*np.sin(self.x[1])
        x2 = x1 + self.l2*np.sin(self.x[2])

        y1 = -self.l1*np.cos(self.x[1])
        y2 = y1-self.l2*np.cos(self.x[2])

        out = [x_m, (x1, y1), (x2, y2), 0, (0,0), (0, 0)]

        if do_vels:
            v_m = self.x[3]
                
            vx1 = v_m + self.l1*np.cos(self.x[1])*self.x[4]
            vx2 = vx1 + self.l2*np.cos(self.x[2])*self.x[5]
    
            vy1 = self.l1*np.sin(self.x[1])*self.x[4]
            vy2 = vy1+self.l2*np.sin(self.x[2])*self.x[5]
    
            out[3] = v_m
            out[4] = (vx1, vy1)
            out[5] = (vx2, vy2)
        return out
        
        
        
            