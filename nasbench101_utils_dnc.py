import numpy as np
import itertools
import copy
from nord.neural_nets import NeuralDescriptor


NUM_LAYERS = 7  # including INPUT and OUTPUT (nasbench-101 supports architectures with up to 7 layers)
NUM_OPS = 3  # nasbench-101 supports three types of operations (CONV1X1, CONV3X3, MAXPOOL3X3)
LENGTH_CONN_SEQ = 0
for i in range(NUM_LAYERS - 1):
    LENGTH_CONN_SEQ += i + 1

MAX_CONNECTIONS = 9  # nasbench-101 requirement

# ops
INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

available_ops = [CONV1X1, CONV3X3, MAXPOOL3X3]
available_ops_onehot = {CONV1X1: [0, 0, 1], CONV3X3: [0, 1, 0], MAXPOOL3X3: [1, 0, 0]}


class Architecture(object):
    def __init__(self, layers, connections, connection_matrix, fitness):
        self.layers = layers
        self.connections = connections
        self.connection_matrix = connection_matrix
        self.fitness = fitness
        self.simplified_layers = copy.deepcopy(layers)
        self.simplified_connection_matrix = copy.deepcopy(connection_matrix)
        self.valid_architecture = True
        self._prune()

    def __str__(self):
        return 'layers: %s, connections: %s, fitness: %s\n connection_matrix:\n %s\n simplified_layers: %s,\n simplified_connection_matrix:\n %s\n valid_architecture: %s\n' % (
            self.layers, self.connections, self.fitness, self.connection_matrix, self.simplified_layers,
            self.simplified_connection_matrix, self.valid_architecture)

    # from https://github.com/fzjcdt/NAS-EA-FA/blob/master/nasbench/lib/model_spec.py
    def _prune(self):
        """Prune the extraneous parts of the graph.

    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """

        num_vertices = np.shape(self.connection_matrix)[0]

        # DFS forward from input
        visited_from_input = set([0])
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if self.connection_matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if self.connection_matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        # If the non-extraneous graph is less than 2 vertices, the input is not
        # connected to the output and the spec is invalid.
        if len(extraneous) > num_vertices - 2:
            self.simplified_connection_matrix = None
            self.simplified_layers = None
            self.valid_architecture = False
            return

        self.simplified_connection_matrix = np.delete(self.simplified_connection_matrix, list(extraneous), axis=0)
        self.simplified_connection_matrix = np.delete(self.simplified_connection_matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            del self.simplified_layers[index]

    def update(self):
        self.connection_matrix = build_connection_matrix(self.connections)
        self.simplified_connection_matrix = copy.deepcopy(self.connection_matrix)
        self.simplified_layers = copy.deepcopy(self.layers)
        self.valid_architecture = True
        self._prune()


def build_connection_matrix(connections):
    connection_matrix = np.zeros((NUM_LAYERS, NUM_LAYERS), dtype=int)
    conn_index = 0
    for row in range(NUM_LAYERS):
        for column in range(row + 1, NUM_LAYERS):
            connection_matrix[row][column] = connections[conn_index]
            conn_index += 1

    return connection_matrix


def get_connections_from_matrix(connection_matrix):
    return connection_matrix[np.triu_indices(connection_matrix.shape[0], k=1)]


def randomly_sample_architecture():
    # initialise random architecture
    layers = [INPUT]
    for _ in range(NUM_LAYERS - 2):
        layers.append(available_ops[np.random.randint(NUM_OPS)])
    layers.append(OUTPUT)

    # form random connections between layers until connections are valid (01 sequence)
    connections = np.array([np.random.randint(2) for _ in range(LENGTH_CONN_SEQ)], dtype=int)

    # initial fitness is 0
    fitness = 0

    architecture = Architecture(layers, connections, build_connection_matrix(connections), fitness)

    return architecture


def create_nord_architecture(architecture):
    # Instantiate a descriptor
    d = NeuralDescriptor()
    index = 0
    for j, layer in enumerate(architecture.simplified_layers):
        index += 1
        d.add_layer(layer, None, 'layer' + str(index))

    for j in range(len(d.layers)):
        for x, node in enumerate(architecture.simplified_connection_matrix[j, :]):
            if node == 1:
                d.connect_layers('layer' + str(j + 1), 'layer' + str(x + 1))

    return d


# def get_node_encoding(layers):
#   node_encoding = []
#   for i in range(1, NUM_LAYERS - 1):
#     node_encoding.append(available_ops_onehot[layers[i]])
#   return np.asarray(node_encoding)

# # node encoding + connections
# def get_sequence(node_encoding, connections):
#   return np.concatenate((node_encoding, connections))

# def get_sequences_distance(s1, s2):
#   dist = 0
#   for i in range(0, 3*(NUM_LAYERS-2), 3):
#     # print(s1[i:i+3], s2[i:i+3])
#     if np.any(s1[i:i+3] != s2[i:i+3]):
#       dist += 1

#   for i in range(3*(NUM_LAYERS-2), len(s1)):
#     if s1[i] != s2[i]:
#       dist += 1

#   return dist

# def get_sequences_distance(s1, s2):
#   dist = 0
#   for bit1, bit2 in zip(s1, s2):
#     if bit1 != bit2:
#       dist += 1
#   return dist

# def get_min_distance(x_train, s):
#   min_d = 100000
#   for x_seq in x_train:
#     min_d = min(min_d, get_sequences_distance(x_seq, s))

#   return min_d

# from official implementation
def get_sequences(ops, matrix) -> list:
    rst = []
    v_num = len(ops)
    for i in range(1, NUM_LAYERS - 1):
        if i < v_num and ops[i] != OUTPUT:
            rst.extend(available_ops_onehot[ops[i]])
        else:
            rst.extend([0, 0, 0])

    for row_index in range(NUM_LAYERS - 1):
        for col_index in range(row_index + 1, NUM_LAYERS):
            if row_index < v_num and col_index < v_num:
                rst.append(matrix[row_index][col_index])
            else:
                rst.append(0)

    return rst


# from official implementation
def get_model_sequences(individual: Architecture) -> list:
    return get_sequences(individual.simplified_layers, individual.simplified_connection_matrix)


# from official implementation
def get_sequence_distance(s1, s2) -> int:
    rst = 0
    for t1, t2 in zip(s1, s2):
        if t1 != t2:
            rst += 1
    return rst


# from official implementation
def get_min_distance(x_train, s):
    min_d = 100000
    for temp_s in x_train:
        min_d = min(min_d, get_sequence_distance(temp_s, s))
    return min_d


def permute_graph(graph, label, permutation):
    """Permutes the graph and labels based on permutation.
  from nasbench.lib.graph_util import permute_graph

  Args:
    graph: np.ndarray adjacency matrix.
    label: list of labels of same length as graph dimensions.
    permutation: a permutation list of ints of same length as graph dimensions.

  Returns:
    np.ndarray where vertex permutation[v] is vertex v from the original graph
  """
    # vertex permutation[v] in new graph is vertex v in the old graph
    forward_perm = zip(permutation, list(range(len(permutation))))
    inverse_perm = [x[1] for x in sorted(forward_perm)]
    edge_fn = lambda x, y: graph[inverse_perm[x], inverse_perm[y]] == 1
    new_matrix = np.fromfunction(np.vectorize(edge_fn),
                                 (len(label), len(label)),
                                 dtype=np.int8)
    new_label = [label[inverse_perm[i]] for i in range(len(label))]
    return new_matrix, new_label


def _label2ops(label):
    ops = []
    for l in label:
        if l == -1:
            ops.append(INPUT)
        elif l == -2:
            ops.append(OUTPUT)
        else:
            ops.append(available_ops[l])

    return ops


def is_upper_triangular(matrix):
    """True if matrix is 0 on diagonal and below."""
    for src in range(np.shape(matrix)[0]):
        for dst in range(0, src + 1):
            if matrix[src, dst] != 0:
                return False

    return True


def get_all_isomorphic_sequences(architecture):
    sequences = []
    connection_matrix = architecture.simplified_connection_matrix
    label = [-1] + [available_ops.index(op) for op in architecture.simplified_layers[1:-1]] + [-2]

    vertices = np.shape(connection_matrix)[0]
    # Note: input and output in our constrained graphs always map to themselves
    # but this script does not enforce that.
    for perm in itertools.permutations(range(1, vertices - 1)):
        full_perm = [0]
        full_perm.extend(perm)
        full_perm.append(vertices - 1)
        pmatrix, plabel = permute_graph(connection_matrix, label, full_perm)
        pmatrix = pmatrix + 0
        ops = _label2ops(plabel)
        if is_upper_triangular(pmatrix) and sum(get_connections_from_matrix(pmatrix)) <= MAX_CONNECTIONS:
            # sequences.append(get_sequence(get_node_encoding(ops).flatten(), get_connections_from_matrix(pmatrix)))
            sequences.append(get_sequences(ops, pmatrix))

    return sequences


def tournament_selection(population, percentage=0.2):
    k = int(len(population) * percentage)
    individual = np.random.choice(population)
    for _ in range(k - 1):
        temp_individual = np.random.choice(population)
        if temp_individual.fitness > individual.fitness:
            individual = temp_individual

    return individual


def bitwise_mutation(individual):
    # layer mutation
    ops_mutation_rate = 1.0 / (len(individual.layers) - 2)
    for i in range(1, len(individual.layers) - 1):
        if np.random.random() < ops_mutation_rate:
            other_ops = []
            for op in available_ops:
                if individual.layers[i] != op:
                    other_ops.append(op)
            individual.layers[i] = np.random.choice(other_ops)

    # connection mutation
    conn_mutation_rate = 1.0 / len(individual.connections)
    temp_connection = copy.deepcopy(individual.connections)
    while True:
        for i in range(len(individual.connections)):
            if np.random.random() < conn_mutation_rate:
                individual.connections[i] = 1 - individual.connections[i]

        individual.update()

        if sum(individual.connections) <= MAX_CONNECTIONS and individual.valid_architecture:
            break
        else:
            individual.connections = copy.deepcopy(temp_connection)

    return individual
