import os


def current_best_val_acc(val_acc, test_acc, naswt_score, best_val_acc, best_test_acc_based_on_val_acc,
                         best_naswt_score_based_on_val_acc):
    if best_val_acc != []:
        if val_acc > best_val_acc[-1]:
            best_naswt_score_based_on_val_acc.append(naswt_score)
            best_val_acc.append(val_acc)
            best_test_acc_based_on_val_acc.append(test_acc)
        else:
            best_naswt_score_based_on_val_acc.append(best_naswt_score_based_on_val_acc[-1])
            best_val_acc.append(best_val_acc[-1])
            best_test_acc_based_on_val_acc.append(best_test_acc_based_on_val_acc[-1])
    else:
        best_naswt_score_based_on_val_acc.append(naswt_score)
        best_val_acc.append(val_acc)
        best_test_acc_based_on_val_acc.append(test_acc)

    return best_val_acc, best_test_acc_based_on_val_acc, best_naswt_score_based_on_val_acc


def current_best_test_acc(test_acc, best_test_acc):
    if best_test_acc != []:
        if test_acc > best_test_acc[-1]:
            best_test_acc.append(test_acc)
        else:
            best_test_acc.append(best_test_acc[-1])
    else:
        best_test_acc.append(test_acc)

    return best_test_acc


def current_best_naswt_score(naswt_score, val_acc, test_acc, best_naswt_score, best_val_acc_based_on_naswt_score,
                             best_test_acc_based_on_naswt_score):
    if best_naswt_score != []:
        if naswt_score > best_naswt_score[-1]:
            best_naswt_score.append(naswt_score)
            best_val_acc_based_on_naswt_score.append(val_acc)
            best_test_acc_based_on_naswt_score.append(test_acc)
        else:
            best_naswt_score.append(best_naswt_score[-1])
            best_val_acc_based_on_naswt_score.append(best_val_acc_based_on_naswt_score[-1])
            best_test_acc_based_on_naswt_score.append(best_test_acc_based_on_naswt_score[-1])
    else:
        best_naswt_score.append(naswt_score)
        best_val_acc_based_on_naswt_score.append(val_acc)
        best_test_acc_based_on_naswt_score.append(test_acc)

    return best_naswt_score, best_val_acc_based_on_naswt_score, best_test_acc_based_on_naswt_score


def current_total_train_time(train_time, total_train_time):
    if total_train_time != []:
        total_train_time.append(total_train_time[-1] + train_time)
    else:
        total_train_time.append(train_time)

    return total_train_time


def current_total_naswt_calc_time(calc_time, total_naswt_calc_time):
    if total_naswt_calc_time != []:
        total_naswt_calc_time.append(total_naswt_calc_time[-1] + calc_time)
    else:
        total_naswt_calc_time.append(calc_time)

    return total_naswt_calc_time


def progress_update(val_acc, test_acc, naswt_score, train_time, naswt_calc_time, best_val_acc,
                    best_test_acc_based_on_val_acc,  best_naswt_score_based_on_val_acc, best_test_acc, best_naswt_score,
                    best_val_acc_based_on_naswt_score, best_test_acc_based_on_naswt_score, train_times,
                    naswt_calc_times, total_train_time, total_naswt_calc_time):
    best_val_acc, best_test_acc_based_on_val_acc, best_naswt_score_based_on_val_acc = \
        current_best_val_acc(val_acc, test_acc, naswt_score, best_val_acc, best_test_acc_based_on_val_acc,
                             best_naswt_score_based_on_val_acc)

    best_test_acc = current_best_test_acc(test_acc, best_test_acc)

    best_naswt_score, best_val_acc_based_on_naswt_score, best_test_acc_based_on_naswt_score = \
        current_best_naswt_score(naswt_score, val_acc, test_acc, best_naswt_score,
                                 best_val_acc_based_on_naswt_score, best_test_acc_based_on_naswt_score)

    train_times.append(train_time)
    naswt_calc_times.append(naswt_calc_time)

    total_train_time = current_total_train_time(train_time, total_train_time)
    total_naswt_calc_time = current_total_naswt_calc_time(naswt_calc_time, total_naswt_calc_time)

    return best_val_acc, best_test_acc_based_on_val_acc, best_naswt_score_based_on_val_acc, best_test_acc, \
           best_naswt_score, best_val_acc_based_on_naswt_score, best_test_acc_based_on_naswt_score, train_times,\
           naswt_calc_times, total_train_time, total_naswt_calc_time


def save_performance(folder_name, exp_repeat_index, start_time, end_time, best_val_acc, best_test_acc_based_on_val_acc,
                     best_naswt_score_based_on_val_acc, best_test_acc, best_naswt_score,
                     best_val_acc_based_on_naswt_score, best_test_acc_based_on_naswt_score, train_times,
                     naswt_calc_times, total_train_time, total_naswt_calc_time):
    with open(os.path.join(folder_name, 'best_val_acc' + str(exp_repeat_index + 1) + '.txt')) as f:
        for element in best_val_acc:
            f.write(str(element) + '\n')

    with open(os.path.join(folder_name, 'best_test_acc_based_on_val_acc' + str(exp_repeat_index + 1) + '.txt')) as f:
        for element in best_test_acc_based_on_val_acc:
            f.write(str(element) + '\n')

    with open(os.path.join(folder_name, 'best_naswt_score_based_on_val_acc' + str(exp_repeat_index + 1) + '.txt')) as f:
        for element in best_naswt_score_based_on_val_acc:
            f.write(str(element) + '\n')

    with open(os.path.join(folder_name, 'best_test_acc' + str(exp_repeat_index + 1) + '.txt')) as f:
        for element in best_test_acc:
            f.write(str(element) + '\n')

    with open(os.path.join(folder_name, 'best_naswt_score' + str(exp_repeat_index + 1) + '.txt')) as f:
        for element in best_naswt_score:
            f.write(str(element) + '\n')

    with open(os.path.join(folder_name, 'best_val_acc_based_on_naswt_score' + str(exp_repeat_index + 1) + '.txt')) as f:
        for element in best_val_acc_based_on_naswt_score:
            f.write(str(element) + '\n')

    with open(os.path.join(folder_name, 'best_test_score_based_on_naswt_score' + str(exp_repeat_index + 1) + '.txt')) as f:
        for element in best_test_acc_based_on_naswt_score:
            f.write(str(element) + '\n')

    with open(os.path.join(folder_name, 'train_times' + str(exp_repeat_index + 1) + '.txt')) as f:
        for element in train_times:
            f.write(str(element) + '\n')

    with open(os.path.join(folder_name, 'total_train_time' + str(exp_repeat_index + 1) + '.txt')) as f:
        for element in total_train_time:
            f.write(str(element) + '\n')

    with open(os.path.join(folder_name, 'naswt_calc_times' + str(exp_repeat_index + 1) + '.txt')) as f:
        for element in naswt_calc_times:
            f.write(str(element) + '\n')

    with open(os.path.join(folder_name, 'total_naswt_calc_time' + str(exp_repeat_index + 1) + '.txt')) as f:
        for element in total_naswt_calc_time:
            f.write(str(element) + '\n')

    with open(os.path.join(folder_name, 'execution_time' + str(exp_repeat_index + 1) + '.txt')) as f:
        f.write(str(end_time - start_time) + '\n')  # in seconds
