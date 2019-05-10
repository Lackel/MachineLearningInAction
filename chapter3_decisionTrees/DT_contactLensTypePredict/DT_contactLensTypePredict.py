import pickle
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

# 计算数据的初始熵
def calshannonent(data_set):
    num_labels = len(data_set)
    label_counts = dict()
    for items in data_set:
        current_label = items[-1]
        label_counts[current_label] = label_counts.get(current_label, 0)+1
    shannon_ent = 0
    for key in label_counts:
        prob = float(label_counts[key])/num_labels
        shannon_ent -= prob*log(prob, 2)
    return shannon_ent


# 划分数据集
def datasetsplit(data_set,axis,value):
    ret_dataset = list()
    for featVet in data_set:
        if featVet[axis] == value:
            reduce_feat_vec = featVet[:axis]
            reduce_feat_vec.extend(featVet[axis+1:])
            ret_dataset.append(reduce_feat_vec)
    return ret_dataset


# 计算最佳分割方式
def choosebestsplit(data_set):
    num_features = len(data_set[0]) - 1
    base_ent = calshannonent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data_set]
        feat_set = set(feat_list)
        new_ent = 0.0
        for value in feat_set:
            sub_dataset = datasetsplit(data_set, i, value)
            prob = float(len(sub_dataset))/len(data_set)
            new_ent += prob*calshannonent(sub_dataset)
        info_gain = base_ent - new_ent
        if info_gain>best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


# 递归构建决策树
def majoritycnt(class_list):
    class_count = dict()
    for vote in class_list:
        class_count[vote] = class_count[vote].get(vote,0) + 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return  sorted_class_count[0][0]


# 创建树
def createtree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    # 类别完全相同则停止划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majoritycnt(class_list)
    best_feat = choosebestsplit(data_set)
    best_label = labels[best_feat]
    my_tree = {best_label:{}}
    del(labels[best_feat])
    feat_values = [example[best_feat] for example in data_set]
    feat_set = set(feat_values)
    for value in feat_set:
        sub_labels = labels[:]
        my_tree[best_label][value] = createtree(datasetsplit(data_set, best_feat, value),sub_labels)
    return my_tree


# 分类判决
def classify(input_tree, feat_labels, test_vec):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    key = test_vec[feat_index]
    feat_value = second_dict[key]
    if isinstance(feat_value, dict):
        class_label = classify(feat_value, feat_labels, test_vec)
    else:
        class_label = feat_value
    return class_label


# 加载数据
def loaddata():
    fr = open('lenses.txt')
    lenses = [example.strip().split('\t') for example in fr.readlines()]
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lenses, lenses_labels


# 存储树
def storetree(my_tree, filename):
    fw = open(filename, 'wb+')
    pickle.dump(my_tree, fw)
    fw.close()


def grabtree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


# data_set, labels = loaddata()
# mytree = createtree(data_set, labels)
# storetree(mytree, 'myfirsttree.txt')
mytree = grabtree('myfirsttree.txt')
data_set, labels = loaddata()
predict = classify(mytree, labels, data_set[1][:])
print(mytree)
print(predict)
