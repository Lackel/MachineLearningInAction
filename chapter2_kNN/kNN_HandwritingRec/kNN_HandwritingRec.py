import numpy as np
import os
import operator


# 数据准备，将输入的图像像素转化为向量
def img2vector(file_name):
    vector = np.zeros((1, 1024))
    fr = open(file_name)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            vector[0, 32*i+j] = int(line[j])
    return vector


# 使用kNN算法计算距离并分类
def knnclassify(inx, dataset, labels, k):
    data_size = dataset.shape[0]
    # 使输入待分类样本维度与样本集维度一致
    diff_mat = np.tile(inx, (data_size, 1)) - dataset
    sq_diff_mat = np.square(diff_mat)
    sq_distance = np.sum(sq_diff_mat, axis=1)
    distacne = sq_distance**0.5
    sorted_distance = distacne.argsort()
    class_count = dict()
    for i in range(k):
        votelabel = labels[sorted_distance[i]]
        class_count[votelabel] = class_count.get(votelabel, 0)+1
    sorted_classcount = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_classcount[0][0]


def handwritinglasstest():
    # 获取训练集
    training_file_list = os.listdir('trainingDigits')
    m_train = len(training_file_list)
    training_mat = np.zeros((m_train, 1024))
    hand_writing_labels = list()
    for i in range(m_train):
        train_file_name = training_file_list[i]  # 0_0.txt
        train_file_str = train_file_name.split('.')[0]  # 0_0
        train_file_num = int(train_file_str.split('_')[0])  # 0
        hand_writing_labels.append(train_file_num)
        training_mat[i, :] = img2vector('trainingDigits/%s' % train_file_name)
    # 获取测试集
    test_file_list = os.listdir('testDigits')
    m_test = len(test_file_list)
    test_mat = np.zeros((m_test, 1024))
    error_num = 0
    print(m_test)
    for i in range(m_test):
        test_file_name = test_file_list[i]
        test_file_str = test_file_name.split('.')[0]
        test_file_num = int(test_file_str.split('_')[0])
        test_mat[i, :] = img2vector('testDigits/%s' % test_file_name)
        predict_num = knnclassify(test_mat[i, :], training_mat, hand_writing_labels, 3)
        # print("the classifier came back with: %d, the real answer is: %d" % (predict_num, test_file_num))
        if predict_num != test_file_num:
            error_num += 1
    print("错误率为：", error_num/m_test)


handwritinglasstest()
