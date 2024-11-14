# The maths behind transforming TSP to Hopfield networks


The representation of TSP using Hopfield networks lends itself to an easy adaptation for the TDTSP later on, and ultimately to get the Hamiltonian of it, as we will see. To leverage this, it makes sense to first explain and optimize some of the formalisms from the HNTSP (Hopfield-Network-TSP) paper [[Hopfield, 1985]](https://www.researchgate.net/publication/19135224_Neural_Computation_of_Decisions_in_Optimization_Problems).

Hopfield networks are directly inspired by the Classical Ising Spin Glass Model, it consists of $n$ nodes on a lattice, the nodes can have values of 0 and 1 (in the Ising Model these values are spin-values of +-1 or +-1/2). The energy function of any HN can be written as:

$
\begin{align}
E(\mathbf{v}) = -\frac{1}{2}\sum_{i = 1}^{n}\sum_{j = 1}^{n} T_{i,j}v_iv_j - \sum_{i=1}^{n}I_iv_i
\end{align}
$

which can be rewritten and vectorized as

$
\begin{align}
E(\mathbf{v}) = -\frac{1}{2}(\mathbf{v}^T T \mathbf{v}) - I^T \mathbf{v}
\end{align}
$

where $\mathbf{v}$ is a vector with all the nodes' states, $T$ is a matrix consisting of the interaction terms between the nodes and $I$ describes the external forces acting on the nodes (network "input"). Note that $(1)$ and $(2)$ immediately strike a resemblance to the QUBO formulation of minimizing the energy landscape of the HN. Indeed, this is what we will leverage later on to quantify the model.

For solving the TSP, Hopfield presents the following $E$-function:

$
\begin{align}
E(\mathbf{v}) = \frac{A}{2} \sum_{X}\sum_{i}\sum_{i \neq j} v_{Xi}v_{Xj} \\
+\frac{B}{2} \sum_{i}\sum_{X}\sum_{X \neq Y} v_{Xi}v_{Yi} \\
+\frac{C}{2} \left[ \left(\sum_{X}\sum_{i} v_{Xi} \right)- n\right]^2 \\
+\frac{D}{2} \sum_{X}\sum_{Y \neq X} d_{XY} \sum_{i} v_{Xi}(v_{Y,i+1} + v_{Y, i-1})
\end{align}$

where $A,B,C,D$ are parameters that need to be adjusted for the model to favor the following conditions. $(3)$ corresponds to inhibitory row connections, meaning there should only be a single 1 in any row. $(4)$ corresponds to inhibitory column connections, so only one 1 in every column. These terms together force the low-energy states of the network to be a valid tour. $(5)$ acts as an additional penalty to ensure that there are only as many 1s in $\mathbf{v}$ as there are cities in the graph. $(6)$ is called the data-term and measures the length of the path represented by $\mathbf{v}$.

Paths are encoded in the following way:

$
\begin{align}
v_{X,i} = 1 \text{ if city X is visited at position i, else } 0
\end{align}
$
Note that $\mathbf{v}$ is a vector, not a matrix, but in $(3,4,5,6)$ the subscripts $X$ and $i$ correspond to city $X$ at position $i$.

The tour A-B-C-A for example would be represented as $[1,0,0,0,1,0,0,0,1]$ (The return to the first site is not denoted in the encoding).

Computing the energy in $(3,4,5,6)$ is very computationally inefficient. One way to tackle this problem is to try and vectorize it and convert it to the standard form $(2)$. The original paper does not offer much improvement, it uses the Kronecker-delta function to describe the implicitly defined $T$-Matrix, which does not lend itself to vectorization using modern computational tools. Instead, one can, after some mathematical magic, come up with these definitions, that are, in fact, equivalent to the original energy function (proofs of these are left as an exercise to the reader haha):

$
\begin{align}
T &= -A*M_1 -B*M_2 - D * M_3 - C \\
M_1 &= I \otimes J \\
M_2 &= \Phi^T(M_1)\Phi \\
M_3 &= A \otimes \alpha
\end{align}
$

where $\otimes$ denotes the Kronecker product, $I$ is the identity matrix, $J$ is the corresponding hollow matrix, $\Phi$ is the square commutator matrix, $A$ the adjacency matrix of the graph on which to solve TSP and $\alpha$, finally, is a circulant matrix, with the first column being 0 everywhere expect in the second and last spot. The matrices live in the following vector spaces: $I \in \mathbb{C}^{n\times n}$, $J \in \mathbb{C}^{n \times n}$, $\Phi \in \mathbb{C}^{n^2 \times n^2}$, $A \in \mathbb{C}^{n \times n}$ and $\alpha \in \mathbb{C}^{n \times n}$. $n$ denotes the number of nodes in the graph. Consequently, $T \in \mathbb{C}^{n^2 \times n^2}$. The linear terms can be written as a vector $K \in \mathbb{C}^{n^2}$ with $2Cn$ everywhere.

Using these, we can rewrite $(3,4,5,6)$ as:

$
\begin{align}
E(\mathbf{v}) = -\frac{1}{2}\mathbf{v}^T T \mathbf{v} - K^T\mathbf{v} + n^2
\end{align}
$

The vectorizations therefore bring our runtime complexity from $O(n^4)$ to $O(n^2)$. The value of the energy function for legal tours is proportional to the path length (and equal if $D=1$), since everything vanishes but the data term. These points on the energy surface correspond to local minima, the global minimum will then be the tour with the shortest length. Also note that the problem of minimizing $(12)$ is now written as a QUBO! So we can map it to the Quantum Ising Model by transforming our binary variables to spin variables and replacing the $v_i$ terms with the Pauli-Z-operator acting on site $i$. We thus get the Ising-Model Hamiltonian, whose Eigenstates with minimum Eigenvalue correspond to the shortest tour. Check the notebooks to try it out!



# Time dependent TSP

To make this problem setting time-dependent, one has to only adjust how to compute $M_3$ since the encoding already holds information about the time at which any site is visited and the edge weights are, in the time-independent case, only repeated.