from nord.neural_nets import BenchmarkEvaluator

import os
import copy
import time
import numpy as np


from params import EXP_REPEAT_TIMES, POPULATION_SIZE, NUM_GEN, T
from nasbench101_utils_dnc import MAX_CONNECTIONS
from nasbench101_utils_dnc import randomly_sample_architecture, create_nord_architecture, tournament_selection, bitwise_mutation

from performance_evaluation import progress_update, save_performance
from save_individual import save_individual_101_dnc


def genetic_algorithm_train_101():
    # Instantiate the evaluator
    evaluator = BenchmarkEvaluator()

    if not os.path.exists('results_ga_dnc101_train'):
        os.mkdir('results_ga_dnc101_train')
    for exp_repeat_index in range(EXP_REPEAT_TIMES):
        start_time = time.time()
        folder_name = os.path.join('results_ga_dnc101_train', 'results' + str(exp_repeat_index + 1))
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        best_val_acc = []
        best_test_acc_based_on_val_acc = []
        train_times = []
        total_train_time = []

        best_test_acc = []

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

                best_val_acc, best_test_acc_based_on_val_acc, best_test_acc, train_times, total_train_time = \
                    progress_update(val_acc=val_acc, test_acc=test_acc, train_time=train_time,
                                    best_val_acc=best_val_acc,
                                    best_test_acc_based_on_val_acc=best_test_acc_based_on_val_acc,
                                    best_test_acc=best_test_acc, train_times=train_times,
                                    total_train_time=total_train_time, fitness='val_acc')

            population = new_population

            with open(os.path.join(folder_name, 'population_epoch' + str(epoch + 1) + '.txt'), 'w') as f:
                ind_num = 0
                for ind in population:
                    ind_num += 1
                    save_individual_101_dnc(f, ind, ind_num, 'val_acc')

            toc = time.time()
            print('experiment index:', exp_repeat_index+1, 'time needed for epoch ' + str(epoch+1) + ':', toc - tic,
                  'sec')

        end_time = time.time()

        save_performance(folder_name, exp_repeat_index, start_time, end_time, best_val_acc,
                         best_test_acc_based_on_val_acc, best_test_acc, train_times, total_train_time,
                         'val_acc')


if __name__ == '__main__':
    np.random.seed(42)
    genetic_algorithm_train_101()
