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
    \lambda_k &= \nabla_{x} H(x_k,u_k,\lambda_{k+1}) = \nabla_x l...
