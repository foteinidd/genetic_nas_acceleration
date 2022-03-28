from nord.neural_nets import BenchmarkEvaluator
import os
import copy
import time

from nasbench101_utils_dnc import EXP_REPEAT_TIMES, POPULATION_SIZE, MAX_CONNECTIONS, NUM_GEN
from nasbench101_utils_dnc import randomly_sample_architecture, create_nord_architecture, tournament_selection, bitwise_mutation


def evolutionary_algorithm():
    # Instantiate the evaluator
    evaluator = BenchmarkEvaluator()

    if not os.path.exists('results_ga_dnc101_train'):
        os.mkdir('results_ga_dnc101_train')
    for exp_repeat_index in range(EXP_REPEAT_TIMES):
        start_time = time.time()
        folder_name = 'results_ga_dnc101_train/results' + str(exp_repeat_index + 1)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        best_val_acc = []
        best_test_acc = []
        train_times = []
        total_time = []

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
        for epoch in range(NUM_GEN*EXP_REPEAT_TIMES):
            tic = time.time()
            new_population = []
            for i in range(POPULATION_SIZE):
                individual = copy.deepcopy(tournament_selection(population))
                new_individual = bitwise_mutation(individual)

                d = create_nord_architecture(new_individual)

                val_acc, train_time = evaluator.descriptor_evaluate(d, acc='validation_accuracy')
                test_acc, train_time = evaluator.descriptor_evaluate(d, acc='test_accuracy')

                new_individual.fitness = val_acc
                new_individual.test_acc = test_acc
                new_individual.train_time = train_time

                new_population.append(new_individual)

                if best_val_acc != []:
                    if val_acc > best_val_acc[-1]:
                        best_val_acc.append(val_acc)
                        best_test_acc.append(test_acc)
                    else:
                        best_val_acc.append(best_val_acc[-1])
                        best_test_acc.append(best_test_acc[-1])
                else:
                    best_val_acc.append(val_acc)
                    best_test_acc.append(test_acc)

                train_times.append(train_time)

                if total_time != []:
                    total_time.append(total_time[-1] + train_time)
                else:
                    total_time.append(train_time)

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
                    f.write('fitness (validation accuracy): ')
                    f.write(str(ind.fitness))
                    f.write('\n')
                    f.write('test accuracy: ')
                    f.write(str(ind.test_acc))
                    f.write('\n')
                    f.write('train time: ')
                    f.write(str(ind.train_time))
                    f.write('\n')

            toc = time.time()
            print('experiment index:', exp_repeat_index+1, 'time needed for epoch ' + str(epoch+1) + ':', toc - tic, 'sec')

        end_time = time.time()

        with open(folder_name + '/best_val_acc' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in best_val_acc:
                f.write(str(element) + '\n')

        with open(folder_name + '/best_test_acc' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in best_test_acc:
                f.write(str(element) + '\n')

        with open(folder_name + '/train_times' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in train_times:
                f.write(str(element) + '\n')

        with open(folder_name + '/total_time' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            for element in total_time:
                f.write(str(element) + '\n')

        with open(folder_name + '/execution_time' + str(exp_repeat_index+1) + '.txt', 'w') as f:
            f.write(str(end_time - start_time) + '\n')  # in seconds


if __name__ == '__main__':
    evolutionary_algorithm()
