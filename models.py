import numpy as np
import networkx as nx
from scipy.linalg import circulant
from collections import OrderedDict
import random

from qiskit.quantum_info import Statevector
from qiskit.result import QuasiDistribution
from qiskit_optimization import QuadraticProgram

# This class initializes the Hopfield Network.
# It should comprise of a method to create the network adjacency
# matrix from the adjacency matrix of the tsp problem and
# (a method to get the energy of the current state)
class HopfieldNetwork:
    def __init__(self, T, I):
        self.T = T
        self.I = I
    
    def compute_energy(self, V, debug=False):
        M1, M2, M3, M4, T, C = self.T
        if debug:
            print(V)
            print("First term (inhibitory row connections):", V.T@M1@V)
            print("First term (inhibitory column connections):", V.T@M2@V)
            print("First term (Data term):", V.T@M3@V)
            print("First term (global inhibitory connections):", V.T@M4@V)
            print("Second term (Magnetic Field term):", V.T.dot(self.I))
            print("Energy (with constant term):", -0.5 * V.T@(T@V) - V.T.dot(self.I) + C*T.shape[0])
            print("Constant term:", C*T.shape[0])
            print("If V is a valid tour then the energy with the constant term is equal to the tour length!")
        return -0.5 * V.T@(T@V) - V.T.dot(self.I)
    def evolve(self, v_0):
        raise NotImplementedError("This method is not yet implemented")

# This class initializes the TSP problem and provides the T and I matrix for the hopfield network
class TSP:
    def __init__(self, G, A=500, B=500, C=20, D=1):
        self.adj = nx.adjacency_matrix(G).toarray()
        self.n = self.adj.shape[0]
        self.I = 2*C * (self.n) * np.ones(self.n**2)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.offset = C * self.n**2
        self.T = self.compute_T()
    #Need the commutation matrix for computing M2 in T
    def commutation_matrix(self, n):
        # Initialize an empty n^2 x n^2 matrix
        K = np.zeros((n * n, n * n))
        # Fill the matrix based on the commutation matrix definition
        for i in range(n):
            for j in range(n):
                # Calculate the row and column indices in K where we put a 1
                row_index = i * n + j
                col_index = j * n + i
                K[row_index, col_index] = 1
        return K
    
    #Need the circulant matrix for computing M4 in T
    def circulant_matrix(self):
        first_column = np.zeros(self.n)
        first_column[1] = 1
        first_column[-1] = 1
        return circulant(first_column)
    
    def compute_T(self):
        Id = np.eye(self.n)
        J = 1-Id
        M1 = np.kron(Id, J)
        Phi = self.commutation_matrix(self.n)
        M2 = Phi.T@(M1@Phi)
        alpha_matrix = self.circulant_matrix()
        M3 = np.kron(self.adj,alpha_matrix)
        M4 = np.ones((self.n**2, self.n**2))
        T = -self.A*M1 -self.B*M2 -self.C*2*M4 - self.D*M3
        return (M1, M2, M3, self.C*2*M4, T, self.C)
    
    # Interpret a hopfield network state as a solution to tsp
    def interpret(self, solution):
        sol = solution.reshape((self.n, self.n))
        # Check the structure first
        rows, cols = np.nonzero(sol)
        for i in range(len(rows)-1):
            if rows[i] in rows[i+1:]:
                print("Non-legal solution!")
                return
        for i in range(len(cols)-1):
            if cols[i] in cols[i+1:]:
                print("Non-legal solution!")
                return
        # Try to interpret the solution by taking the max of each row and clipping all values below to zero
        for i in range(self.n):
            row = sol[i,:]
            max_val = np.max(row)
            row[row<max_val] = 0
            if max_val != 0:
                row[row == max_val] = 1
            sol[i,:] = row
        # Solution is interpreted by taking sol as a permutation matrix (sol.T is sol^{-1})
        return sol.T.dot(np.arange(self.n))
    
    # Get the Hopfield encoding for a tour (mainly used for debugging)
    def get_encoding(self, arr):
        enc = np.zeros((self.n,self.n))
        for i, k in enumerate(arr):
            enc[k,i] = 1
        return enc.flatten()
    
    # Compute the actual tour length corresponding to any hopfield network state
    # HopfieldNetwork.compute_energy does this as well but you have to add Cn**2 
    # as a constant term, which does not fit into the Ising Model formulation.
    def get_cost(self, solution):
        sol = self.interpret(solution)
        tour_length = 0
        for i in range(len(sol)-1):
            tour_length += self.adj[int(sol[i]), int(sol[i+1])]
        tour_length += self.adj[int(sol[-1]), int(sol[0])]
        return tour_length
    
    def to_qubo(self):
        qubo = QuadraticProgram()
        J = self.T[-2]
        for i in range(self.n**2):
            qubo.binary_var("x"+str(i))
        quadratic_terms_dict = {}
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                quadratic_terms_dict[("x"+str(i), "x"+str(j))] = J[i,j]/2
        linear_terms = list(self.I)
        qubo.minimize(constant=-self.offset, linear=linear_terms, quadratic=quadratic_terms_dict)
        return qubo
    
    #From qiskit
    def sample_most_likely(self, state_vector) -> np.ndarray:
        """Compute the most likely binary string from state vector.

        Args:
            state_vector: state vector or counts or quasi-probabilities.

        Returns:
            binary string as numpy.ndarray of ints.

        Raises:
            ValueError: if state_vector is not QuasiDistribution, Statevector,
                np.ndarray, or dict.
        """
        if isinstance(state_vector, QuasiDistribution):
            probabilities = state_vector.binary_probabilities()
            binary_string = max(probabilities.items(), key=lambda kv: kv[1])[0]
            x = np.asarray([int(y) for y in reversed(list(binary_string))])
            return x
        elif isinstance(state_vector, Statevector):
            probabilities = state_vector.probabilities()
            n = state_vector.num_qubits
            k = np.argmax(np.abs(probabilities))
            x = np.zeros(n)
            for i in range(n):
                x[i] = k % 2
                k >>= 1
            return x
        elif isinstance(state_vector, (OrderedDict, dict)):
            # get the binary string with the largest count
            binary_string = max(state_vector.items(), key=lambda kv: kv[1])[0]
            x = np.asarray([int(y) for y in reversed(list(binary_string))])
            return x
        elif isinstance(state_vector, np.ndarray):
            n = int(np.log2(state_vector.shape[0]))
            k = np.argmax(np.abs(state_vector))
            x = np.zeros(n)
            for i in range(n):
                x[i] = k % 2
                k >>= 1
            return x
        else:
            raise ValueError(
                "state vector should be QuasiDistribution, Statevector, ndarray, or dict. "
                f"But it is {type(state_vector)}."
            )
'''UNDOCUMENTED'''
class CompleteGraph:
    def __init__(self, n):
        # Example Graph
        self.G = nx.complete_graph(n)

        for (start, end) in self.G.edges:
            self.G.edges[start, end]['weight'] = random.randint(1, 10)
    def get_graph(self):
        return self.G
    def draw_graph(self):
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.G, seed=8)  # positions for all nodes - seed for reproducibility

        # nodes
        nx.draw_networkx_nodes(self.G, pos, node_size=700)

        # edges
        nx.draw_networkx_edges(self.G, pos, edgelist=self.G.edges, width=5)

        # node labels
        nx.draw_networkx_labels(self.G, pos, font_size=20, font_family="sans-serif")
        # edge weight labels
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    def get_adj(self):
        import networkx as nx
        return nx.adjacency_matrix(self.G).toarray()
'''UNDOCUMENTED'''
class TemporalGraph: #dummy class
    def __init__(self, G_list):
        pass
    def __init__(self, n):
        self.n = n
        graphs = [CompleteGraph(n)]
        self.adj = graphs[-1].get_adj()
        for i in range(1,n):
            graphs.append(CompleteGraph(n))
            new_adj = graphs[-1].get_adj()
            self.adj = np.hstack([self.adj, new_adj])
        self.graphs = graphs
    def draw_graphs(self):
        for g in self.graphs:
            g.draw_graph()
'''UNDOCUMENTED'''
class TDTSP(TSP):
    def __init__(self, temporal_graph, A=500, B=500, C=200, D=1):
        self.adj = temporal_graph.adj
        self.n = temporal_graph.n
        self.I = 2*C * (self.n) * np.ones(self.n**2)
        self.A = A
        self.B = B
        self.C = C
        self.D = 2*D
        self.offset = C * self.n**2
        self.T = self.compute_T()
    def selector(self, i):
        result = np.zeros((self.n, self.n))
        result[i,i] = 1
        return result
    def get_commuter(self, size, idx):
        result = np.zeros(size)
        result[idx] = 1
        return circulant(result).T
    def compute_K(self):
        selector = self.selector(0)
        P_1 = self.get_commuter(self.n,1)
        P_2 = self.get_commuter((self.n)**2, -self.n)
        A = self.adj.copy()
        K = np.kron(self.adj, selector)
        for i in range(1,self.n):
            selector = P_1.T@selector@P_1
            A = A@P_2
            K += np.kron(A, selector)
        return K[:self.n**2, :self.n**2]
    
    def compute_T(self):
        Id = np.eye(self.n)
        J = 1-Id
        M1 = np.kron(Id, J)
        Phi = self.commutation_matrix(self.n)
        M2 = Phi.T@(M1@Phi)
        K = self.compute_K()
        P = self.get_commuter(self.n, -1).T
        M3 = K@(np.kron(Id, P))
        M4 = np.ones((self.n**2, self.n**2))
        T = -self.A*M1 -self.B*M2 -self.C*2*M4 - self.D*M3
        return (M1, M2, M3, self.C*2*M4, T, self.C)