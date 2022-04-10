from nord.neural_nets import BenchmarkEvaluator, NASWT_Evaluator
from nas_101 import ModelSpec, Network
import os
import copy
import time

from nasbench101_utils_dnc import EXP_REPEAT_TIMES, POPULATION_SIZE, MAX_CONNECTIONS, NUM_GEN, T
from nasbench101_utils_dnc import randomly_sample_architecture, create_nord_architecture, tournament_selection, bitwise_mutation

import argparse

parser = argparse.ArgumentParser(description='NASBench')
parser.add_argument('--module_vertices', default=7, type=int, help='#vertices in graph')
parser.add_argument('--max_edges', default=9, type=int, help='max edges in graph')
parser.add_argument('--available_ops', default=['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'],
                    type=list, help='available operations performed on vertex')
parser.add_argument('--stem_out_channels', default=128, type=int, help='output channels of stem convolution')
parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--epochs', default=100, type=int, help='#epochs of training')
parser.add_argument('--learning_rate', default=0.025, type=float, help='base learning rate')
parser.add_argument('--lr_decay_method', default='COSINE_BY_STEP', type=str, help='learning decay method')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 regularization weight')
parser.add_argument('--grad_clip', default=5, type=float, help='gradient clipping')
parser.add_argument('--load_checkpoint', default='', type=str, help='Reload model from checkpoint')
parser.add_argument('--num_labels', default=10, type=int, help='#classes')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')

args = parser.parse_args(args=[])


def evolutionary_algorithm_naswt():
    # Instantiate the evaluators
    evaluator = BenchmarkEvaluator()
    naswt_evaluator = NASWT_Evaluator()

    if not os.path.exists('results_ga_dnc101_naswt_' + str(args.batch_size)):
        os.mkdir('results_ga_dnc101_naswt_' + str(args.batch_size))
    for exp_repeat_index in range(7, EXP_REPEAT_TIMES+1):
        start_time = time.time()
        folder_name = 'results_ga_dnc101_naswt_' + str(args.batch_size) + '/results' + str(exp_repeat_index + 1)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        best_score = []
        best_val_acc = []
        best_test_acc = []
        train_times = []
        naswt_calc_times = []
        total_train_time = []
        total_naswt_calc_time = []

        best_score_absolute = []
        best_val_acc_based_on_fitness = []
        best_test_acc_based_on_fitness = []

        best_test_acc_absolute = []

        # Randomly sample POPULATION_SIZE architectures with an initial fitness of 0
        total_population = []
        for _ in range(POPULATION_SIZE):
            is_valid_architecture = False
            while not is_valid_architecture:
                architecture = randomly_sample_architecture()
                # check if connection number is ok for nasbench-101
                if sum(architecture.connections) <= MAX_CONNECTIONS and architecture.valid_architecture:
                    total_population.append(architecture)
                    is_valid_architecture = True

        population = copy.deepcopy(total_population)

        # evolutionary algorithm
        for epoch in range(NUM_GEN*T):
            num_arch = 0
            tic = time.time()
            new_population = []
            for i in range(POPULATION_SIZE):
                num_arch += 1
                individual = copy.deepcopy(tournament_selection(population))
                new_individual = bitwise_mutation(individual)

                d = create_nord_architecture(new_individual)

                val_acc, train_time = evaluator.descriptor_evaluate(d, acc='validation_accuracy')
                test_acc, train_time = evaluator.descriptor_evaluate(d, acc='test_accuracy')

                arch = ModelSpec(matrix=new_individual.simplified_connection_matrix,
                                 ops=new_individual.simplified_layers)
                net = Network(arch, args)
                K_matrix, score, calc_time = naswt_evaluator.net_evaluate(net=net, batch_size=args.batch_size,
                                                                           dataset=args.dataset)

                new_individual.fitness = score
                new_individual.val_acc = val_acc
                new_individual.test_acc = test_acc
                new_individual.train_time = train_time
                new_individual.naswt_calc_time = calc_time

                print('experiment:', exp_repeat_index + 1, 'epoch:', epoch + 1, 'num_arch:', num_arch, 'naswt_calc_time:', calc_time, 'sec')

                new_population.append(new_individual)

                if best_val_acc != []:
                    if val_acc > best_val_acc[-1]:
                        best_score.append(score)
                        best_val_acc.append(val_acc)
                        best_test_acc.append(test_acc)
                    else:
                        best_score.append(best_score[-1])
                        best_val_acc.append(best_val_acc[-1])
                        best_test_acc.append(best_test_acc[-1])
                else:
                    best_score.append(score)
                    best_val_acc.append(val_acc)
                    best_test_acc.append(test_acc)

                train_times.append(train_time)
                naswt_calc_times.append(calc_time)

                if total_train_time != []:
                    total_train_time.append(total_train_time[-1] + train_time)
                    total_naswt_calc_time.append(total_naswt_calc_time[-1] + calc_time)
                else:
                    total_train_time.append(train_time)
                    total_naswt_calc_time.append(calc_time)

                if best_test_acc_absolute != []:
                    if test_acc > best_test_acc_absolute[-1]:
                        best_test_acc_absolute.append(test_acc)
                    else:
                        best_test_acc_absolute.append(best_test_acc_absolute[-1])
                else:
                    best_test_acc_absolute.append(test_acc)

                if best_score_absolute != []:
                    if score > best_score_absolute[-1]:
                        best_score_absolute.append(score)
                        best_val_acc_based_on_fitness.append(val_acc)
                        best_test_acc_based_on_fitness.append(test_acc)
                    else:
                        best_score_absolute.append(best_score_absolute[-1])
                        best_val_acc_based_on_fitness.append(best_val_acc_based_on_fitness[-1])
                        best_test_acc_based_on_fitness.append(best_test_acc_based_on_fitness[-1])
                else:
                    best_score_absolute.append(score)
                    best_val_acc_based_on_fitness.append(val_acc)
                    best_test_acc_based_on_fitness.append(test_acc)

            population = new_population

            with open(folder_name + '/population_epoch' + str(epoch + 1) + '.txt', 'w') as f:
                ind_num = 0
                for ind in population:
                    ind_num += 1
                    f.write('architecture' + str(ind_num) + '\n')
                    f.write('layers: ')
                    for op in ind.layers:
                        f.write(op + ' ')
                    f.write('\n')
                    f.write('simplified layers: ')
                    for op in ind.simplified_layers:
                        f.write(op + ' ')
                    f.write('\n')
                    f.write('connections: ')
                    for conn in ind.connections:
                        f.write(str(int(conn)) + ' ')
                    f.write('\n')
                    f.write('simplified connection matrix: ')
                    f.write('\n')
                    for row in ind.simplified_connection_matrix:
                        for conn in row:
                            f.write(str(int(conn)) + ' ')
                        f.write('\n')
                    f.write('fitness (naswt score): ')
                    f.write(str(ind.fitness))
                    f.write('\n')
                    f.write('validation accuracy: ')
                    f.write(str(ind.val_acc))
                    f.write('\n')
                    f.write('test accuracy: ')
                    f.write(str(ind.test_acc))
                    f.write('\n')
                    f.write('train time: ')
                    f.write(str(ind.train_time))
                    f.write('\n')
                    f.write('naswt calculation time: ')
                    f.write(str(ind.naswt_calc_time))
                    f.write('\n')

            toc = time.time()
            print('experiment index:', exp_repeat_index+1, 'time needed for epoch ' + str(epoch+1) + ':', toc - tic, 'sec')

        end_time = time.time()

        with open(folder_name + '/best_naswt_score' + str(exp_repeat_index + 1) + '.txt', 'w') as f:
            for element in best_score:
                f.write(str(element) + '\n')

        with open(folder_name + '/best_val_acc' + str(exp_repeat_index + 1) + '.txt', 'w') as f:
            for element in best_val_acc:
                f.write(str(element) + '\n')

        with open(folder_name + '/best_test_acc' + str(exp_repeat_index + 1) + '.txt', 'w') as f:
            for element in best_test_acc:
                f.write(str(element) + '\n')

        with open(folder_name + '/train_times' + str(exp_repeat_index + 1) + '.txt', 'w') as f:
            for element in train_times:
                f.write(str(element) + '\n')

        with open(folder_name + '/total_train_time' + str(exp_repeat_index + 1) + '.txt', 'w') as f:
            for element in total_train_time:
                f.write(str(element) + '\n')

        with open(folder_name + '/naswt_calc_times' + str(exp_repeat_index + 1) + '.txt', 'w') as f:
            for element in naswt_calc_times:
                f.write(str(element) + '\n')

        with open(folder_name + '/total_naswt_calc_time' + str(exp_repeat_index + 1) + '.txt', 'w') as f:
            for element in total_naswt_calc_time:
                f.write(str(element) + '\n')

        with open(folder_name + '/execution_time' + str(exp_repeat_index + 1) + '.txt', 'w') as f:
            f.write(str(end_time - start_time) + '\n')  # in seconds


if __name__ == '__main__':
    evolutionary_algorithm_naswt()
