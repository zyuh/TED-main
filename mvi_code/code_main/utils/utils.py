# coding = utf-8

import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write_csv(file, tag, content):
    """
    写入csv文件

    :param file:
    :param tag: A list of names of per coloumn
    :param content:
    :return:
    """
    with open(file, 'w') as f:
        writer = csv.writer(f)
        if tag:
            writer.writerow(tag)
        writer.writerows(content)


def write_json(file, content):
    """

    :param file: 保存json文件的路径名和文件名
    :param content: dict
    :return:
    """
    with open(file, "w") as f:
        json.dump(content, f)


def draw_ROC(tpr, fpr, best_index, tangent=False, save_path=None):
    plt.title('ROC')
    plt.xlabel('1 - SP')
    plt.ylabel('SE')

    plt.plot(tpr, fpr, color='red', marker='o', mec='r', mfc='r', label='ROC')

    # 切线上的两点
    if tangent:
        x = [0, 1 + fpr[best_index] - tpr[best_index]]
        y = [tpr[best_index] - fpr[best_index], 1]
        plt.plot(x, y, color='blue', marker='*', mec='b', mfc='b')

    plt.grid()
    plt.savefig(save_path)


if __name__ == '__main__':
    write_csv(file='dataset/tmp.csv', tag=[], content=[['123'], ['122']])
