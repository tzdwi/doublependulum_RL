# The Double Pendulum: Can we make it stand on its head with Reinforcement Learning?

Imagine a double pendulum with rigid arms connecting weights of masses $m_1$ and $m_2$, with $m_1$ connected by a massless arm of length $l_1$ to a motorized cart of mass $m_m$, and $m_2$ connected to $m_1$ by a massless arm of length $l_2$.

We can express the positions of the masses $(x_1,y_1)$ and $(x_2, y_2)$ in terms of the position of the cart, $x_m$, and the angles of each of the arms relative to vertical, $\theta_1$ and $\theta_2$ as $x_1 = x_m + l_1\sin\theta_1$, $y_1 = -l_1\cos\theta_1$, $x_2 = x_1 + l_2\sin\theta_2$, and $y_2 = y_1 - l_2\cos\theta_2$. 

With these definitions, we can express a lagrangian, $L(x_m, \dot{x}_m, \theta_1, \dot{\theta}_1, \theta_2, \dot{\theta}_2)$ as:

$L = \frac{1}{2}(m_m+m_1+m_2)\dot{x}_m^2 + m_2\dot{x}_m[(\frac{m_1}{m_2}+1)\dot{\theta}_1l_1\cos\theta_1+\dot{\theta_2}l_2\cos\theta_2] + \frac{1}{2}m_2[(\frac{m_1}{m_2}+1)\dot{\theta}_1^2l_1^2 + \dot{\theta}_2^2l_2^2] +$
$m_2 \dot{\theta}_1\dot{\theta}_2l_1l_2\cos(\theta_1-\theta_2) + gm_2[(\frac{m_1}{m_2}+1)l_1cos\theta_1 + l_2\cos\theta_2] + x_mF_m$

where the final term, $x_mF_m$ represents an external force applied to the motorized cart. Applying the Euler-Lagrange equations, we arrive at the equations of motion, first for $x_m$...

$(m_m + m_1 + m_2)\ddot{x}_m + (m_1+m_2)l_1\cos\theta_1\ddot{\theta}_1 + m_2l_2\cos\theta_2\ddot{\theta}_2-(m_1+m_2)l_1\sin\theta_1\dot{\theta}_1^2 - m_2l_2\sin\theta_2\dot{\theta}_2^2 = F_m$

then $\theta_1$, after lots of arithmetic and cancellations...

$(m_1+m_2)l_1\cos\theta_1\ddot{x}_m+(m_1+m_2)l_1^2\ddot{\theta}_1+m_2l_1l_2\cos(\theta_1-\theta_2)\ddot{\theta}_2+m_2l_1l_2\sin(\theta_1-\theta_2)\dot{\theta}_2^2+g(m_1+m_2)l_1\sin\theta_1=0$

Finally $\theta_2$:

$m_2l_2\cos\theta_2\ddot{x}_m+m_2l_1l_2\cos(\theta_1-\theta_2)\ddot{\theta}_1+m_2l_2^2\ddot{\theta}_2 - m_2l_1l_2\sin(\theta_1-\theta_2)\dot{\theta}_1^2+gm_2l_2\sin\theta_2$

Denoting $\theta = [x_m, \theta_1, \theta_2]^T$, we can write

$\mathbf{D}(\theta)\ddot{\theta}+\mathbf{C}(\theta,\dot{\theta})\dot{\theta} + \mathbf{G}(\theta) = \mathbf{H}F_m$

with two matrix terms:

$$\mathbf{D}(\theta)=
\begin{pmatrix}
m_m+m_1+m_2 & (m_1+m_2)l_1\cos\theta_1 & m_2l_2\cos\theta_2 \\
(m_1+m_2)l_1\cos\theta_1 & (m_1+m_2)l_1^2 & m_2l_1l_2\cos(\theta_1-\theta_2) \\
m_2l_2\cos\theta_2 & m_2l_1l_2\cos(\theta_1-\theta_2) & m_2l_2^2
\end{pmatrix}
$$

(note that this matrix is symmetric and invertible),

$$\mathbf{C}(\theta,\dot{\theta})=
\begin{pmatrix}
0 & -(m_1+m_2)l_1\sin\theta_1\dot{\theta}_1 & -m_2l_2\sin\theta_2\dot{\theta}_2 \\
0 & 0 & m_2l_1l_2\sin(\theta_1-\theta_2)\dot{\theta} \\
0 & -m_2l_1l_2\sin(\theta_1-\theta_2)\dot{\theta}_1 & 0
\end{pmatrix}
$$

(not symmetric, though has some interesting structure), and two vector terms:

$$\mathbf{G}(\theta)=[0,g(m_1+m_2)l_1\sin\theta_1,gm_2l_2\sin\theta_2]^T; \mathbf{H}=[1, 0, 0]^T$$

Defining a state vector $\mathbf{x}\in\\mathbb{R}^6$:

$$\mathbf{x}=(\theta, \dot{\theta})^T$$

the entire system can be written as:

$$\dot{\mathbf{x}}=\begin{pmatrix}
\mathbf{0} & \mathbf{I} \\
\mathbf{0} & -\mathbf{D}^{-1}\mathbf{C} \\
\end{pmatrix}\mathbf{x} + 
\begin{pmatrix}
\mathbf{0} \\
-\mathbf{D}^{-1}\mathbf{G}
\end{pmatrix} +
\begin{pmatrix}
\mathbf{0} \\
-\mathbf{D}^{-1}\mathbf{H}
\end{pmatrix}F_m$$

Now we have something we can work with! But let's give thanks where thanks are due, to A. Bogdanov (2004) https://www.researchgate.net/publication/250107215_Optimal_Control_of_a_Double_Inverted_Pendulum_on_a_Cart
