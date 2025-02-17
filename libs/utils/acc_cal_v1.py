#!/usr/bin/env python

import os
import numpy as np


def topKaccuracy(y_out, y, k):
    L = y.shape[0]

    m = np.ones_like(y, dtype=np.int8)
    lm = np.triu(m, 24)
    mm = np.triu(m, 12)
    sm = np.triu(m, 6)

    avg_pred = (y_out + y_out.transpose((1, 0))) / 2.0
    truth = np.concatenate((avg_pred[..., np.newaxis], y[..., np.newaxis]), axis=-1)

    accs = []
    for x in [lm, mm, sm]:
        selected_truth = truth[x.nonzero()]
        selected_truth_sorted = selected_truth[(selected_truth[:, 0]).argsort()[::-1]]
        tops_num = min(selected_truth_sorted.shape[0], L / k)
        truth_in_pred = selected_truth_sorted[:, 1].astype(np.int8)
        corrects_num = np.bincount(truth_in_pred[0: tops_num], minlength=2)
        acc = 1.0 * corrects_num[1] / (tops_num + 0.0001)
        accs.append(acc)

    return accs


def topLmatrix(predict_matrix):
    m, n = predict_matrix.shape
    points = []
    for i in range(m):
        for j in range(i + 1, n):
            points.append((i, j, predict_matrix[i][j]))
    topL = sorted(points, key=lambda x: x[-1], reverse=True)[: max(m, n)]
    matrix = np.zeros((m, n), dtype=np.uint8)
    for p in topL:
        matrix[p[0]][p[1]] = 1
    return matrix


def evaluate(predict_matrix, contact_matrix):
    acc_k_1 = topKaccuracy(predict_matrix, contact_matrix, 1)
    acc_k_2 = topKaccuracy(predict_matrix, contact_matrix, 2)
    acc_k_5 = topKaccuracy(predict_matrix, contact_matrix, 5)
    acc_k_10 = topKaccuracy(predict_matrix, contact_matrix, 10)
    tmp = []
    tmp.append(acc_k_1)
    tmp.append(acc_k_2)
    tmp.append(acc_k_5)
    tmp.append(acc_k_10)
    return tmp


def output_result(avg_acc):
    print("Long Range:")
    print("Method    L/10         L/5          L/2        L")
    print("Acc :     %.3f        %.3f        %.3f      %.3f" \
          % (avg_acc[3][0], avg_acc[2][0], avg_acc[1][0], avg_acc[0][0]))
    print("Medium Range:")
    print("Method    L/10         L/5          L/2        L")
    print("Acc :     %.3f        %.3f        %.3f      %.3f" \
          % (avg_acc[3][1], avg_acc[2][1], avg_acc[1][1], avg_acc[0][1]))
    print("Short Range:")
    print("Method    L/10         L/5          L/2        L")
    print("Acc :     %.3f        %.3f        %.3f      %.3f" \
          % (avg_acc[3][2], avg_acc[2][2], avg_acc[1][2], avg_acc[0][2]))


def test():
    with open("data/PSICOV/psicov.list", "r") as fin:
        names = [line.rstrip("\n") for line in fin]

    accs = []
    for i in range(len(names)):
        name = names[i]
        print("processing in %d: %s" % (i + 1, name))

        # prediction_path = "data/PSICOV/clm/"
        # prediction_path = "data/PSICOV/ccmpred"
        # prediction_path = "data/PSICOV/psicov_matrix"
        prediction_path = "data/PSICOV/mf_matrix"
        f = os.path.join(prediction_path, name + ".mfDCA")
        if not os.path.exists(f):
            print("not exist...")
            continue
        y_out = np.loadtxt(f)

        dist_path = "data/PSICOV/dis/"
        y = np.loadtxt(os.path.join(dist_path, name + ".dis"))
        y[y > 8] = 0
        y[y != 0] = 1
        y = y.astype(np.int8)
        y = np.tril(y, k=-6) + np.triu(y, k=6)

        acc = evaluate(y_out, y)
        accs.append(acc)
    accs = np.array(accs)
    avg_acc = np.mean(accs, axis=0)
    output_result(avg_acc)
