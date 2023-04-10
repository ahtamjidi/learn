# Convex optimization
- [Convex Optimization](https://www.stat.cmu.edu/~ryantibs/convexopt-S15/)

  - [https://www.stat.cmu.edu/\~ryantibs/convexopt-S15/lectures/24-prox-newton.pdf](https://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/24-prox-newton.pdf)

## Proximal methods
- [Google Drive](https://drive.google.com/drive/search?q=owner:marin%40zoox.com)
- [https://web.stanford.edu/\~boyd/papers/pdf/prox\_algs.pdf](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf)

- [https://www.cse.iitb.ac.in/\~cs709/notes/readingAssignment/ProjectedNewton.pdf](https://www.cse.iitb.ac.in/~cs709/notes/readingAssignment/ProjectedNewton.pdf)

There are several resources available to learn more about stage-based Newton method with projection and related optimization techniques. Here are a few suggestions:

1.  "Numerical Optimization" by Jorge Nocedal and Stephen J. Wright - This is a comprehensive textbook that covers various optimization techniques, including Newton methods and projection methods.
    
2.  "Convex Optimization" by Stephen Boyd and Lieven Vandenberghe - This is another comprehensive textbook that covers convex optimization techniques, including projection methods.
    
3.  "Nonlinear Programming: Concepts, Algorithms, and Applications" by Mokhtar S. Bazaraa, Hanif D. Sherali, and C. M. Shetty - This book provides an introduction to nonlinear programming and covers several optimization techniques, including Newton methods and projection methods.
    
4.  "Large-Scale Nonlinear Optimization" edited by Thomas F. Coleman and Yuying Li - This book provides a collection of papers on large-scale nonlinear optimization, including stage-based Newton method with projection.
    
5.  Research papers - There are numerous research papers available on stage-based Newton method with projection and related optimization techniques. Some examples include "A stage-based inexact Newton method for large-scale nonlinear optimization with simple bounds" by Wen-Xin Zhang and "Projection methods for large-scale constrained optimization" by Marco Campi and Sergiy Vorobyov.
    

These resources should provide a solid foundation for understanding stage-based Newton method with projection and how it can be used to solve constrained nonlinear optimization problems.


## Active set methods

- [Active-set method - Wikipedia](https://en.wikipedia.org/wiki/Active-set_method#:~:text=In%20mathematical%20optimization%2C%20the%20active,a%20simpler%20equality%2Dconstrained%20subproblem.)

**Q:** In general an active-set algorithm has the following structure
??
Find a feasible starting point
repeat until "optimal enough"
solve the equality problem defined by the active set (approximately)
compute the Lagrange multipliers of the active set
remove a subset of the constraints with negative Lagrange multipliers
search for infeasible constraints
end repeat

**Q:** Methods that can be described as active-set methods include:
??
-   [Successive linear programming](https://en.wikipedia.org/wiki/Successive_linear_programming "Successive linear programming") (SLP)
-   [Sequential quadratic programming](https://en.wikipedia.org/wiki/Sequential_quadratic_programming "Sequential quadratic programming") (SQP)
-   [Sequential linear-quadratic programming](https://en.wikipedia.org/wiki/Sequential_linear-quadratic_programming "Sequential linear-quadratic programming") (SLQP)
-   [Reduced gradient method](https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm "Frank–Wolfe algorithm") (RG)
-   [Generalized reduced gradient method](https://en.wikipedia.org/w/index.php?title=Generalized_Reduced_Gradient&action=edit&redlink=1 "Generalized Reduced Gradient (page does not exist)") (GRG)

## Stage wise newton method
- [ACC05.pdf](https://ieor.berkeley.edu/wp-content/uploads/2019/03/ACC05.pdf)
- [web.mit.edu/dimitrib/www/ProjectedNewton.pdf](http://web.mit.edu/dimitrib/www/ProjectedNewton.pdf)



## Primal Dual tricks
Teach me primal dual trick in optimization. Use equations, provide intuition and point me to good references

The primal-dual method is a powerful optimization technique used to solve a variety of problems, including linear programming, convex optimization, and variational inequalities. The basic idea behind the primal-dual method is to solve a pair of related optimization problems simultaneously: the primal problem and its dual problem. In this answer, I will provide an overview of the primal-dual method, along with some equations and intuition to help you understand this optimization technique.

The Primal Problem: The primal problem is the original optimization problem that we want to solve. It is usually formulated as a minimization problem, with some constraints. For example, consider the following linear programming problem:

$$\begin{align} \min_{x} \quad & c^Tx \ \text{s.t.} \quad & Ax \geq b \ & x \geq 0, \end{align}$$

where $x\in\mathbb{R}^n$ is the optimization variable, $c\in\mathbb{R}^n$ is the cost vector, $A\in\mathbb{R}^{m\times n}$ is the constraint matrix, and $b\in\mathbb{R}^m$ is the constraint vector. The constraints in this problem ensure that the optimization variable $x$ satisfies certain requirements.

The Dual Problem: The dual problem is a related optimization problem that is derived from the primal problem. It is usually formulated as a maximization problem, with some constraints. For the linear programming problem above, the dual problem is:

$$\begin{align} \max_{y} \quad & b^Ty \ \text{s.t.} \quad & A^Ty \leq c \ & y \geq 0, \end{align}$$

where $y\in\mathbb{R}^m$ is the dual variable, $b\in\mathbb{R}^m$ is the constraint vector, $A\in\mathbb{R}^{m\times n}$ is the constraint matrix, and $c\in\mathbb{R}^n$ is the cost vector.

The Primal-Dual Method: The primal-dual method solves both the primal and dual problems simultaneously, by updating the primal and dual variables in a coordinated way. The basic idea is to use a Lagrangian function to relate the primal and dual problems. The Lagrangian function is defined as:

$$\begin{equation} L(x,y) = c^Tx - y^T(Ax - b), \end{equation}$$

where $x$ and $y$ are the primal and dual variables, respectively. The primal and dual problems are then related through the Lagrangian function as follows:

$$\begin{align} \min_{x} \max_{y} \quad & L(x,y) \ \text{s.t.} \quad & x \geq 0, \ & y \geq 0. \end{align}$$

The primal-dual method starts by setting some initial values for the primal and dual variables, and then iteratively updates them using the following steps:

$$\begin{align} x_{k+1} &= \operatorname{argmin}_{x} L(x,y_k), \ y_{k+1} &= \operatorname{argmax}_{y} L(x_{k+1},y), \end{align}$$

where $k$ is the iteration number. The update for $x_{k+1}$ is called the primal update, and the update for $y_{k+1}$ is called the dual update. These updates are performed in a coordinated way, and ensure that the primal and dual variables are updated in a way


**Q** What is augmented Lagrangian method of optimization 
??

The augmented Lagrangian method can be written as follows:

$$
\begin{align}
\min_{x} &\quad f(x) \\\
\text{subject to} &\quad g_i(x) \leq 0, \quad i=1,\dots,m \\\
&\quad h_j(x) = 0, \quad j=1,\dots,n
\end{align}
$$
where $f(x)$ is the objective function, $g_i(x)$ are the inequality constraints, $h_j(x)$ are the equality constraints, and $\lambda$ is the penalty parameter.

The augmented Lagrangian function is defined as follows:

$$L_\rho(x,\lambda) = f(x) + \sum_{i=1}^m \rho_i (g_i(x))^2 + \sum_{j=1}^n \lambda_j h_j(x)$$

where $\rho_i$ and $\lambda_j$ are the dual variables associated with the inequality and equality constraints, respectively.

The augmented Lagrangian method is an iterative algorithm. In each iteration, the following steps are performed:

1. Solve the following problem:

$$\min_{x} \quad L_\rho(x,\lambda)$$

This problem can be solved using a standard optimization algorithm, such as Newton's method.

2. Update the dual variables:

$$\lambda_j^{k+1} = \lambda_j^k + \rho_i (g_i(x^k))^2$$

3. Update the penalty parameter:

$$\rho_i^{k+1} = \rho_i^k \theta$$

where $\theta$ is a relaxation parameter.

The augmented Lagrangian method is guaranteed to converge to a stationary point of the original problem. However, the convergence rate can be slow.

Here are some references that you may find helpful:

* Bertsekas, D. P. (1999). Nonlinear programming. Athena Scientific.
* Fletcher, R. (2000). Practical methods of optimization. John Wiley & Sons.
* Nash, S. G., & Sofer, A. (2001). Linear and nonlinear programming. Courier Corporation.
---




The augmented Lagrangian method is an optimization technique that is used to solve constrained optimization problems. It is based on the idea of converting the original constrained optimization problem into an unconstrained optimization problem by adding a penalty term to the objective function that penalizes violations of the constraints.

The augmented Lagrangian function is defined as:

��(�,�)=�(�)+∑�=1���ℎ�(�)+�2∑�=1�ℎ�2(�)Lρ​(x,λ)=f(x)+∑i=1m​λi​hi​(x)+2ρ​∑i=1m​hi2​(x)

where $x$ is the vector of optimization variables, $f(x)$ is the objective function, $h_i(x)$ are the constraint functions, $\lambda_i$ are the Lagrange multipliers, and $\rho$ is a penalty parameter that controls the degree of violation of the constraints.

The augmented Lagrangian method involves iteratively minimizing the augmented Lagrangian function with respect to $x$ and updating the Lagrange multipliers using the following update rule:

���+1=���+�ℎ�(��+1)λik+1​=λik​+ρhi​(xk+1)

where $k$ is the iteration number and $x^{k+1}$ is the updated value of $x$ obtained from the minimization step.

The method is based on the idea of adding a penalty term to the objective function that increases as violations of the constraints become more severe. The penalty parameter $\rho$ controls the degree of violation of the constraints and is typically chosen to increase as the algorithm progresses, to encourage convergence.

The augmented Lagrangian method is a popular technique for solving constrained optimization problems because it is robust and easy to implement. It can handle a wide range of constraint types and can often find solutions that are difficult to obtain using other optimization methods.


\newpage
\section{LECTURE 6}

\section{Agenda}
This lecture covers Pontryagin's Minimum Principle, and an introduction to Linear Quadratic Regulators. 

\section{Pontryagin's Minimum Principle}
\begin{itemize}
    - The Pontryagin's Minimum Principle is also a ``maximum principle'' if instead of minimizing a cost function, we are maximizing a reward function. 
    - Essentially provides first order necessary conditions of deterministic optimal control problems. 
    - In discrete time, it's just a special case of the KKT. 
\end{itemize}
Consider the problem we have before: 
$$
\begin{align}
    \min_{x_{1:N}, u_{1:N-1}} \ & J(x_{1:N}, u_{1:N-1}) = \sum_{k=1}^{N-1} l(x_k, u_k) + l_F(x_N) \\
    \ni \ & x_{n+1} = f(x_n,u_n)
\end{align}
$$
In this setting, we will consider torque limits applied to the problem, but it's hard to handle constraints on state (such as collision constraints, like we had before). 

\noindent
We can form the Lagrangian of this problem as follows: 
$$
\begin{align}
    L = \sum_{k=1}^{N-1} \Big[ l(x_k,u_k) + \lambda_{k+1}^+ (f(x_k,u_k)-x_{k+1}) \Big] + l_F(x_N) \\
\end{align}
$$
This result is usually stated in terms of the ``Hamiltonian'': 
$$
\begin{align}
    H(x, u, \lambda) = l(x,u) + \lambda^T f(x,u)
\end{align}
Plugging in $H$ into the Lagrangian $L$:
$$

$$
\begin{align}
    L = H(x_1, u_1, \lambda_2) + \Big[ \sum_{k=2}^{N-1} H(x_k,u_k,\lambda_{k+1}) - \lambda_k^T x_k \Big] + l_F(x_N) - \lambda_N^T x_N 
\end{align}
$$

Note the change in indexing of the summation.
If we take derivatives with respect to $x$ and $\lambda$: 
$$
\begin{align}
    \frac{\partial L}{\partial \lambda_k} &= \frac{\partial H}{\partial \lambda_k} - x_{k+1} = f(x_k,u_k) - x_{k+1} = 0 \\
    \frac{\partial L}{\partial x_k} &= \frac{\partial H}{\partial x_k} - \lambda_{k}^T = \frac{\partial l}{\partial x_k} + \lambda_{k+1}^T \frac{\partial f}{\partial x_k} - \lambda_k^T = 0 \\
    \frac{\partial L}{\partial x_N} &= \frac{\partial l_F}{\partial x_N} - \lambda_N^T = 0
\end{align}
$$
For $u$, we write the $\min$ explicitly to handle the torque limits:
$$
\begin{align}
    u_k &= \arg\min_{\tilde{u}} H(x_k, \tilde{u}, \lambda_{k+1}) \\
    \ni \ & \tilde{u}  \in \mathcal{U} 
\end{align}
$$
Here, $\tilde{u} \in \mathcal{U}$ is shorthand for ``in feasible set'', for example, 
$u_{\min} \leq \tilde{u} \leq u_{\max}$.\\

\noindent
In summary, we have: 
$$
\begin{align}
    x_{k+1} &= \nabla_{\lambda} H(x_k, u_k, \lambda_{k+1}) = f(x_k, u_k) \\
    \lambda_k &= \nabla_{x} H(x_k,u_k,\lambda_{k+1}) = \nabla_x l(x_k,u_k) + (\frac{\partial f}{\partial x})^T \lambda_{k+1} \\
    u_k &= \arg\min_{\tilde{u}} H(x_k, \tilde{u}, \lambda_{k+1}) \\
    &\ \ \ni \ \tilde{u} \in \mathcal{U} \\
    \lambda_{N} &= \frac{\partial l_F}{\partial X_{N}}
\end{align}
$$
Now these can be stated almost identically in continuous time: 
$$
\begin{align}
    \dot{x} &= \nabla_{\lambda} H(x, u, \lambda) = f_{\rm continuous}(x, u) \\
    \dot{\lambda} &= \nabla_{x} H(x,u,\lambda) = \nabla_x l(x,u) + (\frac{\partial f_{\rm continuous}}{\partial x})^T \lambda \\
    u &= \arg\min_{\tilde{u}} H(x, \tilde{u}, \lambda) \\
    &\ \ \ni \  \tilde{u} \in \mathcal{U} \\
    \lambda_{N} &= \frac{\partial l_F}{\partial X_{N}}
\end{align}
$$
\subsection{Notes}

- Historically, many algorithms were based on forward / backward integration of the continuous ODEs for $x(t), \lambda(t)$ for performing gradient descent on $u(t)$. 
- These are called ``indirect'' and / or ``shooting'' methods. 
- In continuous time, $\lambda(t)$ is called ``co-state'' trajectory. 
- These methods have largely fallen out of favor as computers and solvers have improved. 

\section{Linear Quadratic Regulator (LQR)}
A very common class of controllers is the Linear Quadratic Regulator. In this setting, we have a quadratic cost, and linear dynamics, specified as: 
$$
\begin{align}
    \min_{x_{1:N}, u_{1:N-1}} &\sum_{k=1}^{N-1} \Big[ \frac{1}{2} x_k^T Q x_k + \frac{1}{2} u_k^T R_k u_k  \Big] + \frac{1}{2} x_N^T Q_N x_N \\
    & \ \ni \ x_{k+1} = A_k x_k + B_k u_k 
\end{align}
$$
Here, we have: $Q \geq 0, R \geq 0$. 

- The goal of this problem is to drive the system to the origin.
- It's considered a ``time-invariant'' LQR if $A_k=A, B_k=B, Q_k=Q, R_k=R \ \forall \ k$, and is a ``time-varying'' LQR (TVLQR) otherwise. 
- We typically use time-invariant LQRs for stabilizing an equilibrium, and TVLQR for tracking trajectories.
- Can (locally) approximate many non-linear problems, and are thus very commonly used. 
- There are also many extensions of LQR, including the infinite horizon case, and stochastic LQR. 
- It's been called the ``crown jewel of control theory.''


\subsection{LQR with Indirect Shooting}
Consider: 
$$
\begin{align}
    x_{k+1} &= A x_k + B u_k \\
    \lambda_k &= Q x_k + A^T \lambda_{k+1}, \ \lambda_N = Q x_N \\
    u_{k} &= -R^{-1} B^T \lambda_{k+1} \ \textrm{This gives the gradients of the cost w.r.t.} u_k
\end{align}
$$
The procedure for LQR with indirect shooting is: 
- Start with an initial guess trajectory. 
- Simulate (or ``rollout'') to get $x(t)$.
- Backward pass to get $\lambda(t)$ and $\Delta u(t)$.
- Rollout with line search on $\Delta u$.
- Go to (3) until convergence.


\subsection{Example}
Check out the example of the double integrator in the lecture. Here, we have - 
$$
\begin{align}
    \dot{x} = \begin{bmatrix}
        \dot{q} \\
        \ddot{q} 
    \end{bmatrix}
    = 
    \begin{bmatrix}
        0 \ \ 1  \\
        0 \ \ 0
    \end{bmatrix}
    \begin{bmatrix}
        q \\
        \dot{q} 
    \end{bmatrix}
    + 
    \begin{bmatrix}
        0 \\
        1
    \end{bmatrix}
    u 
\end{align}
$$
Think of this as a ``sliding brick'' on ice, without friction. Here, the discrete-time version of this system is - 

$$
\begin{align}
    x_{k+1} = 
    \begin{bmatrix}
        1 \ \ h  \\
        0 \ \ 1
    \end{bmatrix}
    \begin{bmatrix}
        q_k \\
        \dot{q_k} 
    \end{bmatrix}
    + 
    \begin{bmatrix}
        \frac{1}{2} h^2 \\
        h
    \end{bmatrix}
    u_k 
\end{align} 
$$
where
$$
 \begin{bmatrix}
        1 \ \ h  \\
        0 \ \ 1
    \end{bmatrix}$ represents the $A$ matrix, and $    \begin{bmatrix}
        \frac{1}{2} h^2 \\
        h
    \end{bmatrix}$ represents the $B$ matrix.
$$  
\subsection{Bryson's Rule}


- Cost-tuning heuristic. 
- Set diagonal elements of $Q$ and $R$ to $\frac{1}{ \textrm{max value}^2 }$.
- Normalizes all cost terms to $1$. 
- Good starting point for tuning. 
- In the scalar case, if we have: 
    $$
    \begin{align}
        \frac{1}{2} Q x^2 + \frac{1}{2} R u^2 
    \end{align}
  $$
    Then we can set $Q$ such that $Q x^2 = \frac{1}{x_{\max}^2 } x_{\max}^2 = 1$ , and similarly $R$ such that: $R u^2 = \frac{1}{u_{\max}^2} u_{\max}^2 = 1$


# LECTURE 5

# Agenda

In this lecture, we cover regularization, and line searches with
constraints. We’ll then move on to deterministic optimal control.

# Regularization and Duality

Consider the following problem:
$$\begin{aligned}
    \min_x f(x) \\
    \ni c(x) = 0
\end{aligned}$$
We can express this as:
$$\begin{aligned}
    \min_x f(x) + P\_{\infty} (c(x)) \\
    P\_{\infty} (x) = \begin{cases}
0, & x=0 \\
+ \infty, & x \neq 0 
\end{cases}
\end{aligned}$$
Practically, this is terrible, but we can get the same effect by
solving:
$$\begin{aligned}
    \min_x \max\_{\lambda} f(x) + \lambda^T c(x) 
\end{aligned}$$

Whenever *c*(*x*) ≠ 0, the inner problem gives  + ∞. Similarly for
inequalities:
$$\begin{aligned}
    \min_x \\&f(x) \\
    \ni \\&c(x) \geq 0 \\
    &\implies \\
    \min_x \\&f(x) + P\_{\infty}^+ (c(x))
\end{aligned}$$

Here, we have:
$$\begin{aligned}
    P\_{\infty}^+ (x) &= \begin{cases}
0, & x \geq 0 \\
+ \infty, & x \< 0 
\end{cases}
\implies \\ 
\min_x \max\_{\lambda \geq 0} f(x) - \lambda c(x) 
\end{aligned}$$
where *f*(*x*) − *λ**c*(*x*) is our Lagrangian *L*(*x*,*λ*).  
For convex problems, one can switch the order of the min  and max , and
the solution does not change; this is the notion of “dual problems”!
This isn’t true in general, however, i.e. for non-convex problems.  
The interpretation of this is that the KKT conditions define a saddle
point in the space of (*x*,*λ*).  
The KKT system should have dim (*x*) positive eigenvalues and dim (*λ*)
negative eigenvalues at an optimum. Such a system is known as a
“Quasi-definite” linear system.  

## Take-Away Messages:

When regularizing a KKT system, the lower-right block should be
negative!
$$\begin{aligned}
    \begin{bmatrix}
        H + \alpha I \\\\\\\\C^T \\
        C \\\\\\\\-\alpha I 
    \end{bmatrix}
    \begin{bmatrix}
        \Delta x \\
        \Delta \lambda
    \end{bmatrix}
    = 
    \begin{bmatrix}
        -\nabla_x L \\
        - c(x)
    \end{bmatrix}
    , 
    \alpha \> 0
\end{aligned}$$
This makes the system a quasi-definite one.

Remember to check out the example of this in the lecture video. We
observe that we still overshoot the solution, so we need a line search!

## Merit Functions

How do we a do a line search on a root-finding problem? Consider:
$$\begin{aligned}
    \textrm{find} x\* \\\ni \\c(x\*) = 0
\end{aligned}$$
First, we define a scalar “merit function” *P*(*x*), that measure
distance from a solution. A few standard choices for this merit function
include:
$$\begin{aligned}
    P(x) &= \frac{1}{2} c(x)^T c(x) = \frac{1}{2} \|\| c(x) \|\|\_{2}^2 \\
    \textrm{or} \\\\P(x) &= \|\| c(x) \|\|\_{1} \\\\\\\textrm{Note: Could use any norm}.
\end{aligned}$$
Now we could just do Armijo on *P*(*x*):  
  

*α* = 1  
*α* ← *c**α* *x* ← *α**Δ**x*

  

## Constrained Minimization

How about constrained minimization? Here, we want to come up with an
option that specifies how much we are violating the constraint, as well
as how far off the optimum we are / minimizing the objective funciton.
Consider the problem:
$$\begin{aligned}
    \min_x &f(x) \\
    \ni \\\\&c(x) \geq 0 \\
    \ni \\\\&d(x) = 0 \\
    \\&\\\implies \\
    L(x, \lambda, \mu) &= f(x) - \lambda^T c(x) + \mu^T d(x)
\end{aligned}$$
  
We have lots of options for merit functions. One option is:
$$\begin{aligned}
    P(x, \lambda, \mu) &= \frac{1}{2} \|\| \nabla L(x, \lambda, \mu) \|\|\_2^2 \\
\end{aligned}$$
Here, the term ∇*L*(*x*,*λ*,*μ*) is the KKT residual:
$$\begin{aligned}
    \begin{bmatrix}
        \nabla_x L(x,\lambda,\mu) \\
        \min(0, c(x)) \\
        d(x)
    \end{bmatrix}
\end{aligned}$$
  
However, this isn’t the best option to use, because evaluating the
gradient of the KKT condition is as expensive as the newton step solve
itself.

Another option is:
$$\begin{aligned}
    P(x,\lambda, \mu) &= f(x) + \rho \\\|\| \begin{bmatrix}
        \min (0,c(x)) \\ 
        d(x) 
    \end{bmatrix} \|\|\_1
\end{aligned}$$
Here, *ρ* is a scalar trade off between the objective minimization and
constraint satisfaction. Also remember that any norm works here (in
place of the 1 norm depicted), but using the 1 norm is the most common.
This option gives us flexibility, because we can pick trade off *ρ* -
initially we can set this to be low, to drive us close to the optimum
solution, and when we are close to the optimum, we can increase *ρ* to
ensure we satisfy the constraints.

Yet another option is:
$$\begin{aligned}
    P(x, \lambda, \mu) &= f(x) - \tilde{\lambda}^T c(x) + \tilde{\mu}^T d(x) + \frac{\rho}{2} \|\| \min(0,c(x) \|\|\_2^2 + \frac{\rho}{2} \|\| d(x) \|\|\_2^2 
\end{aligned}$$
which is the augmented Lagrangian itself.

Remember to check out the example in the lecture video!

## Take-Away Messages (from the example)

-   *P*(*x*) based on the KKT residual is expensive.

-   Excessively large penalty weights can cause problems.

-   Augmented Lagrangian methods come with a merit function for free. So
    if we’re using the Augmented Lagrangian to solve the problem, just
    use this as a merit function.

  

## Deterministic Optimal Control

Let’s consider the following control problem:
$$\begin{aligned}
    \min\_{x(t), u(t)} &= J(x(t), u(t)) = \int\_{t_0}^{t_f} L(x(t), u(t)) dt + L\_{F} (x(t_f)) \\
    \ni \\&\dot{x}(t) = f(x(t), u(t)) \\
    \ni \\&\textrm{Any other constraints} 
\end{aligned}$$
Here, we minimize across “state” or “input” trajectories. *J* represents
our cost function, *L*(*x*(*t*),*u*(*t*)) represents our “stage cost”,
*L*<sub>*F*</sub>(*x*(*t*<sub>*f*</sub>)) represents a “terminal cost”,
*ẋ*(*t*) = *f*(*x*(*t*),*u*(*t*)) are the dynamics constraint.  
This is an “infinite dimensional” problem in the sense that an infinite
amount of discrete time control points required to fully specify the
control to be applied.

<figure id="fig:l5f1">
<img src="L5_Images/L51.PNG" />
<figcaption>Deterministic optimal control problem</figcaption>
</figure>

-   The solutions to this problem are open loop trajectories.

-   Now, there are a few control problems with analytic solutions in
    continuous time, but not many.

-   We focus on the discrete-time setting, where we have tractable
    algorithms.

## Discrete Time

Consider a discrete-time version of this problem.
$$\begin{aligned}
    \min\_{x\_{1:N}, u\_{1:N-1}} \\& J(x\_{1:N}, u\_{1:N-1}) = \sum\_{k=1}^{N-1} L(x_k, u_k) + L_F(x_N) \\
    \ni \\& x\_{n+1} = f(x_n,u_n) \\
    \ni \\& u\_{\min} \leq u_k \leq u\_{\max} \\\\\textrm{Torque limits.} \\
    \ni \\& c(x_k) \leq 0 \\\forall k \\\\\textrm{Obstacle / collision constraints.}
\end{aligned}$$

-   This version of the problem is now a finite dimensional problem.

-   Samples *x*<sub>*k*</sub>, *u*<sub>*k*</sub> are often called “knot
    points”.

-   We can convert continuous systems to discrete-time problems using
    integration methods such as the Runge-Kutta method etc.

-   Finally, we can convert back from discrete-time problems to
    continuous problems using interpolation.


# LECTURE 5

# Agenda
In this lecture, we cover regularization, and line searches with constraints. We'll then move on to deterministic optimal control. 

# Regularization and Duality
Consider the following problem: 

$$
\begin{align}
    \min_x f(x) \\
    \ni c(x) = 0
\end{align}
$$
We can express this as: 
$$
\begin{align}
    \min_x f(x) + P_{\infty} (c(x)) \\
    P_{\infty} (x) = \begin{cases}
0, & x=0 \\
+ \infty, & x \neq 0 
\end{cases}
\end{align}
$$
Practically, this is terrible, but we can get the same effect by solving:
$$
\begin{align}
    \min_x \max_{\lambda} f(x) + \lambda^T c(x) 
\end{align}
$$

Whenever $c(x) \neq 0$, the inner problem gives $+\infty$. 
Similarly for inequalities: 
$$
\begin{align}
    \min_x \ &f(x) \\
    \ni \ &c(x) \geq 0 \\
    &\implies \\
    \min_x \ &f(x) + P_{\infty}^+ (c(x))
\end{align}
$$

Here, we have: 
$$
\begin{align}
    P_{\infty}^+ (x) &= \begin{cases}
0, & x \geq 0 \\
+ \infty, & x < 0 
\end{cases}
\implies \\ 
\min_x \max_{\lambda \geq 0} f(x) - \lambda c(x) 
\end{align}
$$
where $f(x) - \lambda c(x)$ is our Lagrangian $L(x,\lambda)$.\\

For convex problems, one can switch the order of the $\min$ and $\max$, and the solution does not change; this is the notion of ``dual problems''! This isn't true in general, however, i.e. for non-convex problems. \\

The interpretation of this is that the KKT conditions define a saddle point in the space of $(x, \lambda)$. \\

The KKT system should have $\dim(x)$ positive eigenvalues and $\dim(\lambda)$ negative eigenvalues at an optimum. Such a system is known as a ``Quasi-definite'' linear system. \\

## Take-Away Messages:

When regularizing a KKT system, the lower-right block should be negative! 
$$
\begin{align}
    \begin{bmatrix}
        H + \alpha I \ \ \ \ C^T \\
        C \ \ \ \ -\alpha I 
    \end{bmatrix}
    \begin{bmatrix}
        \Delta x \\
        \Delta \lambda
    \end{bmatrix}
    = 
    \begin{bmatrix}
        -\nabla_x L \\
        - c(x)
    \end{bmatrix}
    , 
    \alpha > 0
\end{align}
$$
This makes the system a quasi-definite one.

Remember to check out the example of this in the lecture video. We observe that we still overshoot the solution, so we need a line search! 

## Merit Functions
How do we a do a line search on a root-finding problem? Consider: 
$$
\begin{align}
    \textrm{find} x* \ \ni \ c(x*) = 0
\end{align}
$$
First, we define  a scalar ``merit function'' $P(x)$, that measure distance from a solution. 
A few standard choices for this merit function include: 
$$
\begin{align}
    P(x) &= \frac{1}{2} c(x)^T c(x) = \frac{1}{2} || c(x) ||_{2}^2 \\
    \textrm{or} \ \ P(x) &= || c(x) ||_{1} \ \ \ \textrm{Note: Could use any norm}.
\end{align}
$$
Now we could just do Armijo on $P(x)$: 

$$
\begin{algorithm}
	\caption{Armijo Rule on $P(x)$}
	\label{alg:armijo2}
	\begin{algorithmic}[1]	
        \State $\alpha = 1$  \Comment{Step Length}
        \While {$f(x + \alpha \Delta x) > f(x) + b \alpha \nabla f(x) ^T \Delta x$} \\ \Comment{$\alpha \nabla f(x) ^T \Delta x$ is the expected reduction from gradient, and $b$ is the tolerance.}
            \State $\alpha \gets c \alpha$ \Comment{$c$ is a scalar $<1$.}
        \EndWhile
        \State $x \gets \alpha \Delta x$
	\end{algorithmic}
\end{algorithm}
$$

## Constrained Minimization
How about constrained minimization? Here, we want to come up with an option that specifies how much we are violating the constraint, as well as how far off the optimum we are / minimizing the objective funciton. Consider the problem:
$$
\begin{align}
    \min_x &f(x) \\
    \ni \ \ &c(x) \geq 0 \\
    \ni \ \ &d(x) = 0 \\
    \ &\ \implies \\
    L(x, \lambda, \mu) &= f(x) - \lambda^T c(x) + \mu^T d(x)
\end{align}
$$ \\

We have lots of options for merit functions. One option is: 
$$
\begin{align}
    P(x, \lambda, \mu) &= \frac{1}{2} || \nabla L(x, \lambda, \mu) ||_2^2 \\
\end{align}
$$
Here, the term $\nabla L(x, \lambda, \mu)$ is the KKT residual: 
$$
\begin{align}
    \begin{bmatrix}
        \nabla_x L(x,\lambda,\mu) \\
        \min(0, c(x)) \\
        d(x)
    \end{bmatrix}
\end{align}
$$\\
However, this isn't the best option to use, because evaluating the gradient of the KKT condition is as expensive as the newton step solve itself.

Another option is:
$$
\begin{align}
    P(x,\lambda, \mu) &= f(x) + \rho \ || \begin{bmatrix}
        \min (0,c(x)) \\ 
        d(x) 
    \end{bmatrix} ||_1
\end{align}
$$
Here, $\rho$ is a scalar trade off between the objective minimization and constraint satisfaction. Also remember that any norm works here (in place of the $1$ norm depicted), but using the $1$  norm is the most common. 
This option gives us flexibility, because we can pick trade off $\rho$ - initially we can set this to be low, to drive us close to the optimum solution, and when we are close to the optimum, we can increase $\rho$ to ensure we satisfy the constraints. 

Yet another option is: 
$$
\begin{align}
    P(x, \lambda, \mu) &= f(x) - \tilde{\lambda}^T c(x) + \tilde{\mu}^T d(x) + \frac{\rho}{2} || \min(0,c(x) ||_2^2 + \frac{\rho}{2} || d(x) ||_2^2 
\end{align}
$$
which is the augmented Lagrangian itself.

Remember to check out the example in the lecture video! 

## Take-Away Messages (from the example)

- $P(x)$ based on the KKT residual is expensive. 
- Excessively large penalty weights can cause problems. 
- Augmented Lagrangian methods come with a merit function for free. So if we're using the Augmented Lagrangian to solve the problem, just use this as a merit function.

\\

## Deterministic Optimal Control
Let's consider the following control problem:
$$
\begin{align}
    \min_{x(t), u(t)} &= J(x(t), u(t)) = \int_{t_0}^{t_f} L(x(t), u(t)) dt + L_{F} (x(t_f)) \\
    \ni \ &\dot{x}(t) = f(x(t), u(t)) \\
    \ni \ &\textrm{Any other constraints} 
\end{align}
$$
Here, we minimize across ``state'' or ``input'' trajectories. $J$ represents our cost function, $L(x(t), u(t))$ represents our ``stage cost'', $L_{F} (x(t_f))$ represents a ``terminal cost'', $\dot{x}(t) = f(x(t), u(t))$ are the dynamics constraint. \\

This is an ``infinite dimensional'' problem in the sense that an infinite amount of discrete time control points required to fully specify the control to be applied. 
\begin{figure}[h!]
    \centering
    \includegraphics[width=0.4\linewidth]{L5_Images/L51.PNG}
    \caption{Deterministic optimal control problem}
    \label{fig:l5f1}
\end{figure}

- The solutions to this problem are open loop trajectories. 
- Now, there are a few control problems with analytic solutions in continuous time, but not many. 
- We focus on the discrete-time setting, where we have tractable algorithms.

## Discrete Time
Consider a discrete-time version of this problem. 
$$
\begin{align}
    \min_{x_{1:N}, u_{1:N-1}} \ & J(x_{1:N}, u_{1:N-1}) = \sum_{k=1}^{N-1} L(x_k, u_k) + L_F(x_N) \\
    \ni \ & x_{n+1} = f(x_n,u_n) \\
    \ni \ & u_{\min} \leq u_k \leq u_{\max} \ \ \textrm{Torque limits.} \\
    \ni \ & c(x_k) \leq 0 \ \forall k \ \ \textrm{Obstacle / collision constraints.}
\end{align}
$$

- This version of the problem is now a finite dimensional problem.
- Samples $x_k,u_k$ are often called ``knot points''. 
- We can convert continuous systems to discrete-time problems using integration methods such as the Runge-Kutta method etc.
- Finally, we can convert back from discrete-time problems to continuous problems using interpolation. 
  



## Resources 
Descent methods and line search: first Wolfe condition 
- https://www.youtube.com/watch?v=X4Pjd-1R-jI
- [Gradient descent and quasi-Newton methods](https://zigavirk.gitlab.io/main.pdf)
- [Textbook: Nonlinear Programming](http://www.athenasc.com/nonlinbook.html)
- [web.mit.edu/dimitrib/www/publ.html](http://web.mit.edu/dimitrib/www/publ.html)
- [GitHub - amkatrutsa/optimization\_course: A course on Optimization Methods](https://github.com/amkatrutsa/optimization_course)
- [https://www.mit.edu/\~dimitrib/lagr\_mult.html](https://www.mit.edu/~dimitrib/lagr_mult.html)
- [Optimal-Control-16-745 · GitHub](https://github.com/Optimal-Control-16-745/)
