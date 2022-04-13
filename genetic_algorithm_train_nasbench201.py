from nord.neural_nets.natsbench_evaluator import NATSBench_Evaluator
import os
import copy
import time


from params import EXP_REPEAT_TIMES, POPULATION_SIZE, NUM_GEN, T
from nasbench201_utils_dnc import randomly_sample_architecture, create_nord_architecture, tournament_selection, bitwise_mutation

from performance_evaluation import progress_update, save_performance
from save_individual import save_individual_201_dnc


def genetic_algorithm_train_201():
    # Instantiate the evaluator
    natsbench_evaluator = NATSBench_Evaluator()

    if not os.path.exists('results_ga_dnc201_train'):
        os.mkdir('results_ga_dnc201_train')
    for exp_repeat_index in range(EXP_REPEAT_TIMES):
        start_time = time.time()
        folder_name = os.path.join('results_ga_dnc201_train', 'results' + str(exp_repeat_index + 1))
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        best_val_acc = []
        best_test_acc = []
        train_times = []
        total_train_time = []

        # Randomly sample POPULATION_SIZE architectures with an initial fitness of 0
        total_population = []
        for _ in range(POPULATION_SIZE):
            is_valid_architecture = False
            while not is_valid_architecture:
                architecture = randomly_sample_architecture()
                if architecture.valid_architecture:

                    d = create_nord_architecture(architecture)

                    # evaluate architecture
                    invalid_nas201 = False
                    try:
                        val_acc, test_acc, train_time = natsbench_evaluator.descriptor_evaluate(d, metrics=[
                            'validation_accuracy',
                            'test_accuracy',
                            'time_cost'])
                    except ValueError:
                        # print('Invalid architecture (not added in population)')
                        # print(d)
                        # print(natsbench_evaluator._descriptor_to_nasnet(d))
                        invalid_nas201 = True

                    if not invalid_nas201:
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

                # evaluate architecture
                train_loss, val_loss, test_loss, train_acc, val_acc, test_acc, latency, train_time = \
                    natsbench_evaluator.descriptor_evaluate(d, metrics=['train_loss',
                                                                        'validation_loss',
                                                                        'test_loss',
                                                                        'train_accuracy',
                                                                        'validation_accuracy',
                                                                        'test_accuracy',
                                                                        'latency',
                                                                        'time_cost'])

                new_individual.fitness = val_acc
                new_individual.test_acc = test_acc
                new_individual.train_time = train_time

                new_individual.train_loss = train_loss
                new_individual.val_loss = val_loss
                new_individual.test_loss = test_loss
                new_individual.train_acc = train_acc
                new_individual.latency = latency

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
                    save_individual_201_dnc(f, ind, ind_num, 'val_acc')

            toc = time.time()
            print('experiment index:', exp_repeat_index+1, 'time needed for epoch ' + str(epoch+1) + ':', toc - tic,
                  'sec')

        end_time = time.time()

        save_performance(folder_name, exp_repeat_index, start_time, end_time, best_val_acc,
                         best_test_acc_based_on_val_acc, best_test_acc, train_times, total_train_time,
                         'val_acc')


if __name__ == '__main__':
    genetic_algorithm_train_201()
