from nord.neural_nets.natsbench_evaluator import NATSBench_Evaluator
from nord.neural_nets import NASWT_Evaluator
import os
import copy
import time

from params import EXP_REPEAT_TIMES, POPULATION_SIZE, NUM_GEN, T
from nasbench201_utils_dnc import randomly_sample_architecture, create_nord_architecture, tournament_selection, \
    bitwise_mutation

from performance_evaluation import progress_update, save_performance

from save_individual import save_individual_201_dnc

from nord.utils import DATA_ROOT
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net


def genetic_algorithm_naswt_201():
    # Initialise NATS-Bench API
    NATSBENCH_NAME = "NATS-tss-v1_0-3ffb9-simple"
    NATSBENCH_TFRECORD = os.path.join(DATA_ROOT, NATSBENCH_NAME)
    filepath = NATSBENCH_TFRECORD
    api = create(os.path.join(filepath, NATSBENCH_NAME), "tss", fast_mode=True, verbose=False)

    # Instantiate the evaluators
    natsbench_evaluator = NATSBench_Evaluator()
    naswt_evaluator = NASWT_Evaluator()

    # NASWT config
    batch_size = 32
    dataset = 'cifar10'

    if not os.path.exists('results_ga_dnc201_naswt_' + str(batch_size)):
        os.mkdir('results_ga_dnc201_naswt_' + str(batch_size))
    for exp_repeat_index in range(EXP_REPEAT_TIMES):
        start_time = time.time()
        folder_name = os.path.join('results_ga_dnc201_naswt_' + str(batch_size), 'results' + str(exp_repeat_index + 1))
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        best_val_acc = []
        best_test_acc_based_on_val_acc = []
        best_naswt_score_based_on_val_acc = []
        train_times = []
        naswt_calc_times = []
        total_train_time = []
        total_naswt_calc_time = []

        best_naswt_score = []
        best_val_acc_based_on_naswt_score = []
        best_test_acc_based_on_naswt_score = []

        best_test_acc = []

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
        for epoch in range(NUM_GEN * T):
            num_arch = 0
            tic = time.time()
            new_population = []
            for i in range(POPULATION_SIZE):
                num_arch += 1
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

                nasnet_arch = natsbench_evaluator._descriptor_to_nasnet(d)
                natsbench_arch_index = api.query_index_by_arch(nasnet_arch)
                config = api.get_net_config(natsbench_arch_index, dataset + '-valid')
                net = get_cell_based_tiny_net(config)
                K_matrix, naswt_score, naswt_calc_time = naswt_evaluator.net_evaluate(net=net, batch_size=batch_size,
                                                                                      dataset=dataset)

                new_individual.fitness = naswt_score
                new_individual.val_acc = val_acc
                new_individual.test_acc = test_acc
                new_individual.train_time = train_time
                new_individual.naswt_calc_time = naswt_calc_time

                new_individual.train_loss = train_loss
                new_individual.val_loss = val_loss
                new_individual.test_loss = test_loss
                new_individual.train_acc = train_acc
                new_individual.latency = latency

                print('experiment:', exp_repeat_index + 1, 'epoch:', epoch + 1, 'num_arch:', num_arch,
                      'naswt_calc_time:', naswt_calc_time, 'sec')

                new_population.append(new_individual)

                best_val_acc, best_test_acc_based_on_val_acc, best_naswt_score_based_on_val_acc, best_test_acc, \
                best_naswt_score, best_val_acc_based_on_naswt_score, best_test_acc_based_on_naswt_score, train_times, \
                naswt_calc_times, total_train_time, total_naswt_calc_time = \
                    progress_update(val_acc=val_acc, test_acc=test_acc, train_time=train_time, best_val_acc=best_val_acc,
                                    best_test_acc_based_on_val_acc=best_test_acc_based_on_val_acc,
                                    best_test_acc=best_test_acc, train_times=train_times,
                                    total_train_time=total_train_time, fitness='naswt', naswt_score=naswt_score,
                                    naswt_calc_time=naswt_calc_time,
                                    best_naswt_score_based_on_val_acc=best_naswt_score_based_on_val_acc,
                                    best_naswt_score=best_naswt_score,
                                    best_val_acc_based_on_naswt_score=best_val_acc_based_on_naswt_score,
                                    best_test_acc_based_on_naswt_score=best_test_acc_based_on_naswt_score,
                                    naswt_calc_times=naswt_calc_times, total_naswt_calc_time=total_naswt_calc_time)

            population = new_population

            with open(os.path.join(folder_name, 'population_epoch' + str(epoch + 1) + '.txt'), 'w') as f:
                ind_num = 0
                for ind in new_population:
                    ind_num += 1
                    save_individual_201_dnc(f, ind, ind_num)

            toc = time.time()
            print('experiment index:', exp_repeat_index + 1, 'time needed for epoch ' + str(epoch + 1) + ':', toc - tic,
                  'sec')

        end_time = time.time()

        save_performance(folder_name, exp_repeat_index, start_time, end_time, best_val_acc,
                         best_test_acc_based_on_val_acc, best_test_acc, train_times, total_train_time,
                         'naswt', best_naswt_score_based_on_val_acc, best_naswt_score,
                         best_val_acc_based_on_naswt_score, best_test_acc_based_on_naswt_score,
                         naswt_calc_times, total_naswt_calc_time)


if __name__ == '__main__':
    genetic_algorithm_naswt_201()
