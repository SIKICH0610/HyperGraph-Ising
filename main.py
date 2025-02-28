import hypernetx as hnx
import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import copy

class UniformHypergraph:
    def __init__(self, num_nodes, node_degree, edge_size):
        self.num_nodes = num_nodes
        self.node_degree = node_degree
        self.edge_size = edge_size
        # self.beta = beta  # Interaction strength
        # self.h = h  # External magnetic field
        self.hypergraph, self.initial_node_values, self.adjacency_dict, self.edge_dict = self._create_uniform_hypergraph()
        # self.energy_history = []

    def _create_uniform_hypergraph(self):
        # Calculate the number of hyperedges needed
        num_edges = (self.num_nodes * self.node_degree) // self.edge_size

        if (self.num_nodes * self.node_degree) % self.edge_size != 0:
            raise ValueError("It's not possible to create a hypergraph with the given parameters.")

        nodes = [f'v{i}' for i in range(1, self.num_nodes + 1)]
        edges = {f'e{j}': [] for j in range(1, num_edges + 1)}

        # Create a list of all node-edge pairs ensuring no self-edges
        node_edge_pairs = []
        for node in nodes:
            node_edge_pairs.extend([(node, i) for i in range(self.node_degree)])

        # Shuffle the pairs to distribute them randomly
        random.shuffle(node_edge_pairs)

        # Assign nodes to edges ensuring each node is added once per edge
        edge_index = 0
        edge_assignment = {edge: set() for edge in edges}

        for node, _ in node_edge_pairs:
            # Find the next available edge for this node that doesn't already contain the node
            while node in edge_assignment[f'e{edge_index + 1}'] or len(edge_assignment[f'e{edge_index + 1}']) >= self.edge_size:
                edge_index = (edge_index + 1) % num_edges

            edge_assignment[f'e{edge_index + 1}'].add(node)
            edges[f'e{edge_index + 1}'] = list(edge_assignment[f'e{edge_index + 1}'])

        # Verify all nodes have the required degree
        node_degrees = {node: 0 for node in nodes}
        for edge in edges:
            for node in edges[edge]:
                node_degrees[node] += 1

        # for node, degree in node_degrees.items():
        #     if degree != self.node_degree:
        #         raise ValueError("Failed to create a uniform hypergraph with the given parameters.")
        #
        # for edge, nodes in edges.items():
        #     if len(nodes) != self.edge_size:
        #         raise ValueError("Failed to create a uniform hypergraph with the given parameters.")

        # Create the hypergraph
        H = hnx.Hypergraph(edges)

        # Assign +1 or -1 values to the nodes
        node_values = {node: random.choice([-1, 1]) for node in H.nodes}

        # Create adjacency dictionary
        adjacency_dict = {node: set() for node in H.nodes}
        for edge, nodes in edges.items():
            for node in nodes:
                adjacency_dict[node].update(set(nodes) - {node})

        return H, node_values, adjacency_dict, edges

    def visualize(self):
        hnx.draw(self.hypergraph)
        plt.show()

    # def get_hamiltonian(self, node):
    #     adjacent_nodes = self.adjacency_dict[node]
    #     interaction_term = sum(self.node_values[node] * self.node_values[adj_node] for adj_node in adjacent_nodes)
    #     magnetic_field_term = self.node_values[node]
    #     hamiltonian = -self.beta * interaction_term - self.h * magnetic_field_term
    #     return hamiltonian

    # def total_energy(self):
    #     total_energy = 0
    #     for node in self.hypergraph.nodes:
    #         total_energy += self.get_hamiltonian(node)
    #     return total_energy / 2  # Each pair counted twice

    # def glauber_dynamics(self, max_steps, energy_repeat_threshold=-1):
    #     self.energy_history = []
    #     num_repeated_terms = 0
    #     last_energy = 0
    #     epsilon = 1e-6
    #     for step in range(max_steps):
    #         node = random.choice(list(self.hypergraph.nodes))
    #         delta_energy = -2 * self.get_hamiltonian(node)
    #         probability = 1 / (1 + math.exp(delta_energy))
    #         if random.random() < probability:
    #             self.node_values[node] *= -1
    #         current_energy = self.total_energy()
    #         self.energy_history.append(current_energy)
    #         if np.linalg.norm(last_energy - current_energy) < epsilon:
    #             num_repeated_terms += 1
    #         # Reset the number of repeated terms to 0
    #         else:
    #             num_repeated_terms = 0
    #
    #         # Check if energy_repeat_threshold is set to -1
    #         if energy_repeat_threshold != -1:
    #             # Check if any energy value has appeared more than the threshold
    #             if num_repeated_terms > energy_repeat_threshold:
    #                 print(
    #                     f"Energy {current_energy} has appeared more than {energy_repeat_threshold} times. Terminating early.")
    #                 break
    #         last_energy = current_energy
    #
    #         if step % 100 == 0:  # Print progress every 100 steps
    #             print(f'Step {step}, Total Energy: {self.energy_history[-1]}')

    def run_simulation(self, beta, h, max_steps, energy_repeat_threshold=-1):
        node_values = copy.deepcopy(self.initial_node_values)
        energy_history = []
        num_repeated_terms = 0
        last_energy = 0
        epsilon = 1e-6

        def get_hamiltonian(node):
            adjacent_nodes = self.adjacency_dict[node]
            interaction_term = sum(node_values[node] * node_values[adj_node] for adj_node in adjacent_nodes)
            magnetic_field_term = node_values[node]
            hamiltonian = -beta * interaction_term - h * magnetic_field_term
            return hamiltonian

        def total_energy():
            total_energy = 0
            for node in self.hypergraph.nodes:
                total_energy += get_hamiltonian(node)
            return total_energy / 2  # Each pair counted twice

        for step in range(max_steps):
            node = random.choice(list(self.hypergraph.nodes))
            delta_energy = -2 * get_hamiltonian(node)
            probability = 1 / (1 + math.exp(delta_energy))
            if random.random() < probability:
                node_values[node] *= -1
            current_energy = total_energy()
            energy_history.append(current_energy)
            if np.linalg.norm(last_energy - current_energy) < epsilon:
                num_repeated_terms += 1
            else:
                num_repeated_terms = 0

            if energy_repeat_threshold != -1 and num_repeated_terms > energy_repeat_threshold:
                print(
                    f"Energy {current_energy} has appeared more than {energy_repeat_threshold} times. Terminating early.")
                break

            last_energy = current_energy

            if step % 100 == 0:
                print(f'Step {step}, Total Energy: {energy_history[-1]}')

        return energy_history

    def find_lowest_energy_step(self, energy_history):
        min_energy = min(energy_history)
        min_step = energy_history.index(min_energy)
        return min_step, min_energy

    def plot_energy_history(self, energy_history, scale='linear'):
        min_step, min_energy = self.find_lowest_energy_step()
        plt.plot(energy_history, label='Total Energy')
        plt.axvline(x=min_step, color='r', linestyle='--', label=f'Lowest Energy at Step {min_step}')
        plt.scatter(min_step, min_energy, color='r')
        plt.text(min_step, min_energy, f'{min_energy}', horizontalalignment='right')
        plt.xlabel('Step')
        plt.ylabel('Total Energy')
        plt.title('Energy Change in the System Over Time')
        plt.yscale(scale)
        plt.legend()
        plt.show()

    def plot_mixing_times(self, num_nodes_list, beta_values, h, max_steps, energy_repeat_threshold, plt_obj):
        mixing_times = {beta: [] for beta in beta_values}

        for num_nodes in num_nodes_list:
            print(f"Running simulation for graph size: {num_nodes}")
            uhg = UniformHypergraph(num_nodes, self.node_degree, self.edge_size)
            for beta in beta_values:
                print(f"  with beta = {beta}")
                energy_history = uhg.run_simulation(beta, h, max_steps, energy_repeat_threshold)
                min_step, min_energy = uhg.find_lowest_energy_step(energy_history)
                mixing_times[beta].append((num_nodes, min_step))
                print(f"  Graph size: {num_nodes}, Mixing time (steps to lowest energy): {min_step}")

        for beta, times in mixing_times.items():
            sizes = [t[0] for t in times]
            steps = [t[1] for t in times]
            plt_obj.plot(sizes, steps, marker='o', label=f'Beta = {beta}')

        plt_obj.xlabel('Graph Size (Number of Nodes)')
        plt_obj.ylabel('Mixing Time (Steps to Lowest Energy)')
        plt_obj.title('Mixing Time vs. Graph Size for Different Beta Values')
        plt_obj.yscale('linear')
        plt_obj.legend()
        plt_obj.grid(True)

    def __str__(self):
        return f'Edges: {self.hypergraph.edges}\nNodes: {self.hypergraph.nodes}\nNode Values: {self.node_values}'


# # Simulation parameters
# num_nodes_list = list(range(20, 501, 10))
# node_degree = 2  # Degree of each node
# edge_size = 4    # Size of each hyperedge
# beta_values = [1.5, 0.75, 0.5, 0.25]  # Interaction strengths
# h = 0   # External magnetic field
# max_steps = 15000  # Maximum number of steps for Glauber dynamics
# energy_repeat_threshold = 50  # Threshold for repeated energy termination
#
# # Create an instance of UniformHypergraph with any initial parameters (they will be overridden in the loop)
# uhg = UniformHypergraph(10, node_degree, edge_size)
#
# # Plot the mixing times for both beta values
# plt.figure(figsize=(10, 10))
# uhg.plot_mixing_times(num_nodes_list, beta_values, h, max_steps, energy_repeat_threshold, plt)
# plt.show()