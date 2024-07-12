import pandas as pd
import matplotlib.pyplot as plt
from main import UniformHypergraph
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

num_nodes_list = list(range(20, 301, 10))  # Graph sizes from 20 to 300 in steps of 20
node_degree = 2  # Degree of each node
edge_size = 5    # Size of each hyperedge
beta = 0.5  # Interaction strength
h = 0.0    # External magnetic field
max_steps = 20000  # Maximum number of steps for Glauber dynamics
num_simulations = 3  # Number of simulations per graph size

results = []

# Run the simulation for different graph sizes
for num_nodes in num_nodes_list:
    for sim in range(num_simulations):
        logger.info(f"Running simulation {sim+1}/{num_simulations} for graph size: {num_nodes}")
        uhg = UniformHypergraph(num_nodes, node_degree, edge_size, beta, h)
        uhg.glauber_dynamics(max_steps, energy_repeat_threshold=100)
        min_step, min_energy = uhg.find_lowest_energy_step()
        results.append((num_nodes, sim+1, min_step))
        logger.info(f"Graph size: {num_nodes}, Simulation: {sim+1}, Mixing time (steps to lowest energy): {min_step}")

df = pd.DataFrame(results, columns=['Graph Size', 'Simulation', 'Mixing Time'])
df.to_csv('mixing_times.csv', index=False)
logger.info("Results saved to mixing_times.csv")

df = pd.read_csv('mixing_times.csv')

mean_mixing_times = df.groupby('Graph Size')['Mixing Time'].mean()
std_mixing_times = df.groupby('Graph Size')['Mixing Time'].std()

plt.figure(figsize=(10, 8))
plt.errorbar(mean_mixing_times.index, mean_mixing_times, yerr=std_mixing_times, fmt='o', capsize=5, label='Mean Mixing Time with Std Dev')
plt.plot(mean_mixing_times.index, mean_mixing_times, linestyle='-', color='r')
plt.xlabel('Graph Size (Number of Nodes)')
plt.ylabel('Mixing Time (Steps to Lowest Energy)')
plt.title('Mixing Time vs. Graph Size with Error Bars')
plt.legend()
plt.grid(True)
plt.show()