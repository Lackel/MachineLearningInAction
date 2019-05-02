import numpy as np
import matplotlib.pyplot as plt
import operator


# 将文本导入并转化为numpy数组格式返回
def file2matrix(filename):
    # 导入文件
    fname = open(filename)
    # 按行读取文件内容
    lines = fname.readlines()
    # 返回行数，即样本数
    num_of_lines = len(lines)
    # 构建输入的样本及其标签
    x_data = np.zeros((num_of_lines, 3), dtype=np.float32)
    y_label = list()
    index = 0
    # 遍历文件，将数据转换成numpy数组格式返回
    for line in lines:
        line = line.strip()  # 去掉空格
        list_from_line = line.split('\t')  # 将数据分割成一个列表
        x_data[index, :] = list_from_line[0:3]
        y_label.append(list_from_line[-1])
        index += 1
    return x_data, y_label


# 数据可视化，观察并分析数据形式，便于数据预处理
def figureplt(x_data, y_label):
    plt.figure()
    # 彩色效果
    # 矩阵第二列和第三列属性来展示数据
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel("每周消费的冰淇淋公升数")
    plt.ylabel("玩视频游戏所耗时间百分比")
    plt.scatter(x_data[:, 1], x_data[:, 2], c=np.squeeze(y_label))
    plt.show()


# 输入数据尺度差异较大，而我们认为三个特征同样重要，因此要进行归一化
def autonorm(x_data):
    min_data = x_data.min(0)
    max_data = x_data.max(0)
    # print(mindata, maxdata)
    average = max_data - min_data
    norm_data = (x_data - min_data)/average
    return norm_data, min_data, average


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


# 利用上述函数进行分类测试
def knntest():
    data, label = file2matrix('datingTestSet2.txt')
    test_rate = 0.10
    norm_data, min_data, average = autonorm(data)
    m = norm_data.shape[0]
    num_test_data = int(test_rate*m)
    num_error = 0
    for i in range(num_test_data):
        prdict_label = knnclassify(norm_data[i, :], norm_data[num_test_data:, :], label[num_test_data:], 3)
        if prdict_label != label[i]:
            num_error += 1
    print("错误率为：", num_error/num_test_data)


# 构建完整系统，输入为用户的三个指标，输出为预测结果
def classifyperson():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("你大概花费百分之多少的时间打游戏？"))
    ff_miles = float(input("你的飞行里程数是多少？"))
    ice_cream = float(input("你每周吃多少公升的冰激凌？"))
    data, label = file2matrix('datingTestSet2.txt')
    norm_data, min_data, average = autonorm(data)
    inx = np.array((ff_miles, percent_tats, ice_cream))
    result = int(knnclassify((inx-min_data)/average, norm_data, label, 3))
    print("She likes you", result_list[result-1])


knntest()


'''while True:
    letter = input("欢迎使用心动值评分系统，快来测试吧！按任意键继续，q退出")
    if letter is 'q':
        break
    else:
        classifyperson()
'''
