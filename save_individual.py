def save_individual_101_dnc(f, ind, ind_num):
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


def save_individual_201_dnc(f, ind, ind_num):
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
    f.write('train loss: ')
    f.write(str(ind.train_loss))
    f.write('\n')
    f.write('validation loss: ')
    f.write(str(ind.val_loss))
    f.write('\n')
    f.write('test loss: ')
    f.write(str(ind.test_loss))
    f.write('\n')
    f.write('train accuracy: ')
    f.write(str(ind.train_acc))
    f.write('\n')
    f.write('latency: ')
    f.write(str(ind.latency))
    f.write('\n')


def save_individual_fitness_approximation(f, ind, ind_num, fitness_score):
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
    f.write('fitness (approximate ' + fitness_score + ')')
    f.write(str(ind.fitness))
    f.write('\n')
