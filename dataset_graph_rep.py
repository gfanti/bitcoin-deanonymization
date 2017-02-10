# data_graph_rep.py
# contains the classes for storing and processing spreading on a data-provided graph

import networkx as nx
import random
# import matplotlib.pyplot as plt
import numpy as np


class DataGraph(nx.Graph):

	def __init__(self, filename, spreading_time = None):
		super(DataGraph, self).__init__(nx.read_gexf(filename))
		self.lambda1 = 1 # spreading rate over the diffusion graph
		
		
		# Read graph and label nodes from 1 to N
		mapping = {}
		for (idx, node) in zip(range(nx.number_of_nodes(self)), self.nodes()):
			mapping[node] = idx
		nx.relabel_nodes(self, mapping, copy=False)

		self.spreading_time = spreading_time	

# Run diffusion over a provided gexf graph
class DataGraphDiffusion(DataGraph):

	def __init__(self, filename, spreading_time = None):
		''' NB: Here the spreading_time	is actually the number of rings of the graph to infect'''
		super(DataGraphDiffusion, self).__init__(filename, spreading_time)
		

		
	def spread_message(self, source = 0, first_spy_only = False, num_corrupt_cnx = 1):
		'''first_spy_only denotes whether this diffusion spread will only be used
		to measure the first spy adversary. In that case, some time-saving optimizations
		can be implemented. Most of the time, this flag will be set to false.'''
		
		self.source = source
		self.num_corrupt_cnx = num_corrupt_cnx

		# Empty the observed timestamps
		self.adversary_timestamps = {} 		# dictionary of adversary report time indexed by node
		self.received_timestamps = {}		# dictionary of message receipt time indexed by node

		# INitialize the process
		current_time = 0
		self.received_timestamps[self.source] = 0
		self.adversary_timestamps[self.source] = self.send_to_adversary(self.source, num_corrupt_cnx)
		if first_spy_only:
			stopping_time = min(stopping_time, self.adversary_timestamps[self.source])
		# self.active = [source]
		self.infected = [source]

		stopping_time = self.spreading_time
		
		self.infected_by_source = {}
		
		self.active = [(source, n) for n in self.neighbors(source)]	# number of active edges
		count = 0
		while self.active:

			# Compute the delay
			node, neighbor = random.choice(self.active)	# the edge that will fire next
			current_time = self.exponential_delay(current_time, self.lambda1 * len(self.active))	# associated spreading delay
			count += 1
			if current_time > stopping_time:
				break
			self.received_timestamps[neighbor] = current_time

			# Mark neighbor as infected
			# print 'Order: ', node, ' infects ', neighbor
			self.infected += [neighbor]
			if node == source:
				self.infected_by_source[neighbor] = True
			else:
				self.infected_by_source[neighbor] = False

			# Find the reporting time
			adversary_timestamp = self.send_to_adversary(neighbor, num_corrupt_cnx)
			if adversary_timestamp <= self.spreading_time:
				self.adversary_timestamps[neighbor] = adversary_timestamp

			# Clean up the list of edges
			self.active.remove((node, neighbor))
			self.active += [(neighbor, n) for n in self.neighbors(neighbor) if n not in self.infected]
			new_boundary = [edge for edge in self.active if edge[0] in self.infected and edge[1] not in self.infected]
			self.active = [i for i in new_boundary]
		# print 'num infected nodes: ', len(self.infected)


		# print 'infected nodes', self.infected, len(self.infected)
		# print 'rx timetsamps', [(n,self.received_timestamps[n]) for n in self.infected]
		# print 'timetsamps', [(n,self.adversary_timestamps[n]) for n in self.infected if n in self.adversary_timestamps]

	def exponential_delay(self, current_time, rate):
		return current_time + np.random.exponential(1.0 / rate)

	def send_to_adversary(self, node, num_corrupt_cnx):
		return self.received_timestamps[node] + np.random.exponential(1.0 / num_corrupt_cnx)
		# return self.received_timestamps[node]

# Run diffusion over a provided gexf graph
class DataGraphTrickle(DataGraph):

	def __init__(self, filename, spreading_time = None):
		''' NB: Here the spreading_time	is actually the number of rings of the graph to infect'''
		super(DataGraphTrickle, self).__init__(filename, spreading_time)	
		

		
	def spread_message(self, source = 0, first_spy_only = False, num_corrupt_cnx = 1):
		'''first_spy_only denotes whether this diffusion spread will only be used
		to measure the first spy adversary. In that case, some time-saving optimizations
		can be implemented. Most of the time, this flag will be set to false.'''
		
		count = 0
		self.source = source

		adversaries = [-(i+1) for i in range(num_corrupt_cnx)]

		# Empty the observed timestamps
		self.adversary_timestamps = {} 		# dictionary of adversary report time indexed by node
		self.received_timestamps = {}		# dictionary of message receipt time indexed by node

		# Initialize the process
		self.received_timestamps[self.source] = 0
		self.active = [source]
		self.infected = [source]

		stopping_time = self.spreading_time
		

		while self.active and count < stopping_time:
			count += 1
			# cycle through the active nodes, and spread with an exponential clock
			for node in self.active:

				uninfected_neighbors = [neighbor for neighbor in self.neighbors(node) if neighbor not in self.infected]
				uninfected_neighbors += adversaries
				# print 'uninfected_neighbors', uninfected_neighbors, adversaries

				# random permutation of neighbors
				ordering = list(np.random.permutation(uninfected_neighbors))
				# print 'ordering', ordering
				signs = [item >= 0 for item in ordering]
				# print 'signs', signs

				# find the reporting time for node
				self.adversary_timestamps[node] = signs.index(False) + 1 + self.received_timestamps[node]

				if first_spy_only and (node == source):
					stopping_time = min(stopping_time, self.adversary_timestamps[node])
					# print 'stopping_time', stopping_time

				# assign the received timestamps for the other nodes
				for idx in range(len(ordering)):
					neighbor = ordering[idx]
					# if the node at time slot t is not a spy
					if neighbor >= 0:
						rx_timestamp = self.received_timestamps[node] + 1 + idx
						self.received_timestamps[neighbor] = rx_timestamp
						self.infected.append(neighbor)
						if rx_timestamp < stopping_time:
							self.active.append(neighbor)
				self.active.remove(node)
				
		# print 'infected nodes', self.infected, len(self.infected)
		# print 'rx timetsamps', [(n,self.received_timestamps[n]) for n in self.infected]
		# print 'timetsamps', [(n,self.adversary_timestamps[n]) for n in self.infected if n in self.adversary_timestamps]