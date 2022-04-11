from nord.neural_nets.natsbench_evaluator import NATSBench_Evaluator
from nord.neural_nets import NeuralDescriptor
import itertools
import numpy as np
import copy
import traceback


NUM_LAYERS = 8  # including INPUT and OUTPUT (nasbench-201 supports architectures with up to 8 layers)
NUM_OPS = 4  # nasbench-101 supports four types of operations (CONV1X1, CONV3X3, AVGPOOL3X3, SKIPCONNECT)
LENGTH_CONN_SEQ = 0
for i in range(NUM_LAYERS - 1):
    LENGTH_CONN_SEQ += i + 1

# MAX_CONNECTIONS = 9  # nasbench-101 requirement

# ops
INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'nor_conv_1x1'
CONV3X3 = 'nor_conv_3x3'
AVGPOOL3X3 = 'avg_pool_3x3'
SKIPCONNECT = 'skip_connect'

available_ops = [CONV1X1, CONV3X3, AVGPOOL3X3, SKIPCONNECT]
available_ops_onehot = {CONV1X1: [0, 0, 0, 1], CONV3X3: [0, 0, 1, 0], AVGPOOL3X3: [0, 1, 0, 0],
                        SKIPCONNECT: [1, 0, 0, 0]}

# Instantiate the evaluator
evaluator = NATSBench_Evaluator()


class Architecture(object):
    def __init__(self, layers, connections, connection_matrix, fitness):
        self.layers = layers
        self.connections = connections
        self.connection_matrix = connection_matrix
        self.fitness = fitness

    def __str__(self):
        return 'layers: %s, connections: %s, fitness: %s\n connection_matrix:\n %s' % (
            self.layers, self.connections, self.fitness, self.connection_matrix)


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


def form_necessary_connections(connection_matrix, connections, layers):
    new_connection_made = False
    for i, _ in enumerate(layers):
        if 0 < i < NUM_LAYERS - 1 and not np.any(connection_matrix[i, :]):
            connection_matrix[i, NUM_LAYERS - 1] = 1
            new_connection_made = True
        elif i == 0 and not np.any(connection_matrix[i, :]):
            connection_matrix[i, np.random.randint(1, NUM_LAYERS)] = 1
            new_connection_made = True

        if 0 < i < NUM_LAYERS - 1 and not np.any(connection_matrix[:, i]):
            connection_matrix[0, i] = 1
            new_connection_made = True
        elif i == NUM_LAYERS - 1 and not np.any(connection_matrix[:, i]):
            connection_matrix[np.random.randint(0, NUM_LAYERS - 1), i] = 1

        if new_connection_made:
            connections = get_connections_from_matrix(connection_matrix)

    return connections


def randomly_sample_architecture():
    # initialise random architecture
    layers = [INPUT]
    for _ in range(NUM_LAYERS - 2):
        layers.append(available_ops[np.random.randint(NUM_OPS)])
    layers.append(OUTPUT)

    # form random connections between layers until connections are valid (01 sequence)
    connections = np.array([np.random.randint(2) for _ in range(LENGTH_CONN_SEQ)], dtype=int)
    connection_matrix = build_connection_matrix(connections)

    # # form necessary connections (in case some layers have no input or no output connections)
    connections = form_necessary_connections(connection_matrix, connections, layers)

    # initial fitness is 0
    fitness = 0

    architecture = Architecture(layers, connections, build_connection_matrix(connections), fitness)

    return architecture


def create_nord_architecture(architecture):
    # Instantiate a descriptor
    d = NeuralDescriptor()
    index = 0
    for j, layer in enumerate(architecture.layers):
        index += 1
        d.add_layer(layer, None, 'layer' + str(index))

    for j in range(len(d.layers)):
        for x, node in enumerate(architecture.connection_matrix[j, :]):
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
            rst.extend([0, 0, 0, 0])

    for row_index in range(NUM_LAYERS - 1):
        for col_index in range(row_index + 1, NUM_LAYERS):
            if row_index < v_num and col_index < v_num:
                rst.append(matrix[row_index][col_index])
            else:
                rst.append(0)

    return rst


# from official implementation
def get_model_sequences(individual: Architecture) -> list:
    return get_sequences(individual.layers, individual.connection_matrix)


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
    connection_matrix = architecture.connection_matrix
    label = [-1] + [available_ops.index(op) for op in architecture.layers[1:-1]] + [-2]

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
        # if is_upper_triangular(pmatrix) and sum(get_connections_from_matrix(pmatrix)) <= MAX_CONNECTIONS:
        if is_upper_triangular(pmatrix):
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
    ops_mutation_rate = 1.0 / (NUM_LAYERS - 2)
    for i in range(1, NUM_LAYERS - 1):
        if np.random.random() < ops_mutation_rate:
            other_ops = []
            for op in available_ops:
                if individual.layers[i] != op:
                    other_ops.append(op)
            individual.layers[i] = np.random.choice(other_ops)

    # connection mutation
    conn_mutation_rate = 1.0 / LENGTH_CONN_SEQ
    temp_connection = copy.deepcopy(individual.connections)
    while True:
        for i in range(LENGTH_CONN_SEQ):
            if np.random.random() < conn_mutation_rate:
                individual.connections[i] = 1 - individual.connections[i]

        connection_matrix = build_connection_matrix(individual.connections)
        individual.connections = form_necessary_connections(connection_matrix, individual.connections,
                                                            individual.layers)

        # if sum(individual.connections) <= MAX_CONNECTIONS:
        #     break
        # else:
        #     individual.connections = copy.deepcopy(temp_connection)

        d = create_nord_architecture(individual)

        # evaluate architecture
        invalid_architecture = False
        try:
            val_acc, test_acc, train_time = evaluator.descriptor_evaluate(d, metrics=['validation_accuracy',
                                                                                      'test_accuracy', 'time_cost'])
        except ValueError:
            print('Invalid architecture (created by mutation)')
            print(d)
            print(evaluator._descriptor_to_nasnet(d))
            invalid_architecture = True
        except:
            print('Exception found (due to mutation)')
            print(traceback.format_exc())
            print(d)
            invalid_nas201 = True

        if not invalid_architecture:
            break
        else:
            individual.connections = copy.deepcopy(temp_connection)

    return individual
