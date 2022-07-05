from nord.neural_nets import BenchmarkEvaluator, NASWT_Evaluator
from nas_101 import ModelSpec, Network

import os
import copy
import time
import numpy as np

from params import EXP_REPEAT_TIMES, POPULATION_SIZE, NUM_GEN, T
from nasbench101_utils_dnc import MAX_CONNECTIONS
from nasbench101_utils_dnc import randomly_sample_architecture, create_nord_architecture, tournament_selection, bitwise_mutation

from performance_evaluation import progress_update, save_performance
from save_individual import save_individual_101_dnc

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


def genetic_algorithm_naswt_101():
    # Instantiate the evaluators
    evaluator = BenchmarkEvaluator()
    naswt_evaluator = NASWT_Evaluator()

    if not os.path.exists('results_ga_dnc101_naswt_' + str(args.batch_size)):
        os.mkdir('results_ga_dnc101_naswt_' + str(args.batch_size))
    for exp_repeat_index in range(EXP_REPEAT_TIMES):
        start_time = time.time()
        folder_name = os.path.join('results_ga_dnc101_naswt_' + str(args.batch_size), 'results' +
                                   str(exp_repeat_index + 1))
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
                K_matrix, naswt_score, naswt_calc_time = naswt_evaluator.net_evaluate(net=net,
                                                                                      batch_size=args.batch_size,
                                                                                      dataset=args.dataset)

                new_individual.fitness = naswt_score
                new_individual.val_acc = val_acc
                new_individual.test_acc = test_acc
                new_individual.train_time = train_time
                new_individual.naswt_calc_time = naswt_calc_time

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
                for ind in population:
                    ind_num += 1
                    save_individual_101_dnc(f, ind, ind_num, 'naswt')

            toc = time.time()
            print('experiment index:', exp_repeat_index+1, 'time needed for epoch ' + str(epoch+1) + ':', toc - tic,
                  'sec')

        end_time = time.time()

        save_performance(folder_name, exp_repeat_index, start_time, end_time, best_val_acc,
                         best_test_acc_based_on_val_acc, best_test_acc, train_times, total_train_time,
                         'naswt', best_naswt_score_based_on_val_acc, best_naswt_score,
                         best_val_acc_based_on_naswt_score, best_test_acc_based_on_naswt_score,
                         naswt_calc_times, total_naswt_calc_time)


if __name__ == '__main__':
    np.random.seed(42)
    genetic_algorithm_naswt_101()
