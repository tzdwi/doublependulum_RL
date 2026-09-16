# The Double Pendulum: Can we make it stand on its head with Reinforcement Learning?

Imagine a double pendulum with rigid arms connecting weights of masses $m_1$ and $m_2$, with $m_1$ connected by a massless arm of length $l_1$ to a motorized cart of mass $m_m$, and $m_2$ connected to $m_1$ by a massless arm of length $l_2$.

We can express the positions of the masses $(x_1,y_1)$ and $(x_2, y_2)$ in terms of the position of the cart, $x_m$, and the angles of each of the arms relative to vertical, $\theta_1$ and $\theta_2$ as $x_1 = x_m + l_1\sin\theta_1$, $y_1 = -l_1\cos\theta_1$, $x_2 = x_1 + l_2\sin\theta_2$, and $y_2 = y_1 - l_2\cos\theta_2$. 

With these definitions, we can express a lagrangian, $L(x_m, \dot{x}_m, \theta_1, \dot{\theta}_1, \theta_2, \dot{\theta}_2)$ as:

$L = \frac{1}{2}(m_m+m_1+m_2)\dot{x}_m^2 + m_2\dot{x}_m[(\frac{m_1}{m_2}+1)\dot{\theta}_1l_1\cos\theta_1+\dot{\theta_2}l_2\cos\theta_2] + \frac{1}{2}m_2[(\frac{m_1}{m_2}+1)\dot{\theta}_1^2l_1^2 + \dot{\theta}_2^2l_2^2] + m_2 \dot{\theta}_1\dot{\theta}_2l_1l_2\cos(\theta_1-\theta_2) + gm_2[(\frac{m_1}{m_2}+1)l_1cos\theta_1 + l_2\cos\theta_2] + x_mF_m$

where the final term, $x_mF_m$ represents an external force applied to the motorized cart. Applying the Euler-Lagrange equations, we arrive at the equations of motion, first for $x_m$...

$(m_m + m_1 + m_2)\ddot{x}_m + m_2[(\frac{m_1}{m_2}+1)(\ddot{\theta}_1l_1\cos\theta_1+\dot{\theta}_1^2\sin\theta_1)+\ddot{\theta}_2l_2\cos\theta_2+\dot{\theta}_2^2\sin\theta_2] = F_m$

then $\theta_1$, after lots of arithmetic and cancellations...

$\ddot{x}_ml_1\cos\theta_1 + \ddot{\theta}_1l_1^2 + \frac{m_2}{m_1+m_2}l_1l_2[\ddot{\theta}_1\cos(\theta_1-\theta_2)+\dot{\theta}_2^2\sin(\theta_1-\theta_2)] + gl_1\sin\theta_1 = 0$

Finally $\theta_2$:


