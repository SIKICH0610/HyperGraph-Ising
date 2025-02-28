import pandas as pd
import matplotlib.pyplot as plt
from main import UniformHypergraph
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

num_nodes_list = list(range(20, 1001, 10))
node_degree = 2  # Degree of each node
edge_size = 5    # Size of each hyperedge
beta_value = 1.5 # Interaction strength
h = 0.0    # External magnetic field
max_steps = 30000  # Maximum number of steps for Glauber dynamics
num_simulations = 30

results = []

# Run the simulation for different graph sizes
for num_nodes in num_nodes_list:
    uhg = UniformHypergraph(num_nodes, node_degree, edge_size)
    for sim in range(num_simulations):
        logger.info(f"Running simulation {sim+1}/{num_simulations} for graph size: {num_nodes}")
        energy_history = uhg.run_simulation(beta_value, h, max_steps, energy_repeat_threshold=50)
        min_step, min_energy = uhg.find_lowest_energy_step(energy_history)
        results.append((num_nodes, sim+1, min_step))
        logger.info(f"Graph size: {num_nodes}, Simulation: {sim+1}, Mixing time (steps to lowest energy): {min_step}")

# Create DataFrame from results
df = pd.DataFrame(results, columns=['Graph Size', 'Simulation', 'Mixing Time'])

# Calculate mean and standard deviation of mixing times
mean_mixing_times = df.groupby('Graph Size')['Mixing Time'].mean()
std_mixing_times = df.groupby('Graph Size')['Mixing Time'].std()

# Plot the results
plt.figure(figsize=(10, 8))
plt.errorbar(mean_mixing_times.index, mean_mixing_times, yerr=std_mixing_times, fmt='o', capsize=5, label='Mean Mixing Time with Std Dev')
plt.plot(mean_mixing_times.index, mean_mixing_times, linestyle='-', color='r')
plt.xlabel('Graph Size (Number of Nodes)')
plt.ylabel('Mixing Time (Steps to Lowest Energy)')
plt.title('Mixing Time vs. Graph Size with Error Bars')
plt.legend()
plt.grid(True)
plt.show()
