#!/usr/bin/env python
# coding=utf-8

import numpy as np
import os
from sklearn import metrics
import time
from models.machines_data_preprocessing import get_data
import shutil


########################
# NB
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x,
              train_y)
    return model


# KNN Classifier
# K最近邻
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
# 逻辑回归
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
# 随机森林
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
# 决策树
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
# 梯度推进
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
# 支持向量机
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
# 支持向量机 交叉验证
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


maxlen = 1024 * 5  # 一条句子最长长度
test_len_rate = 0.1

########## 准备数据
data_path = r'./data/'
save_path = r'./res-machines/'
save_data_path_selected_data_path = save_path
if not os.path.exists(save_data_path_selected_data_path):
    os.mkdir(save_data_path_selected_data_path)
file_str = "handled_reviews.csv"
file_dir = os.path.join(data_path, file_str)

return_str, (x_train_text, train_y), (x_test_text, test_y) = get_data(file_dir,
                                                                      save_data_path_selected_data_path,
                                                                      maxlen=maxlen,
                                                                      test_len_rate=test_len_rate)

train_x = x_train_text
test_x = x_test_text

test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'GBDT', 'SVM']
# test_classifiers = ['NB', 'LR', 'SVM', 'GBDT']
classifiers = {'NB': naive_bayes_classifier,  # 朴素贝叶斯
               'KNN': knn_classifier,  # K最近邻
               'LR': logistic_regression_classifier,  # 逻辑回归
               'RF': random_forest_classifier,  # 随机森林
               'DT': decision_tree_classifier,  # 决策树
               'SVM': svm_classifier,  # 支持向量机
               # 'SVMCV': svm_cross_validation,  # 支持向量机 交叉验证
               'GBDT': gradient_boosting_classifier  # 梯度推进
               }

result_buffer = ''
num_train, dim_train = train_x.shape
num_test, dim_test = test_x.shape
is_binary_class = (len(np.unique(train_y)) == 2)
result_buffer += "\n--------------------\n" + "///" + file_str + "\n"
print('******************** Data Info *********************')
result_buffer += ('******************** Data Info *********************' + '\n')
print('#training data: %d, #testing_data: %d, train_dimension: %d, test_dimension: %d' % (
num_train, num_test, dim_train, dim_test))
result_buffer += (
    '#training data: %d, #testing_data: %d, train_dimension: %d, test_dimension: %d' % (
    num_train, num_test, dim_train, dim_test) + '\n')
result_buffer += return_str
########################
for classifier in test_classifiers:
    print('******************* %s ********************' % classifier)
    result_buffer += ('******************* %s ********************' % classifier + '\n')
    start_time = time.time()
    model = classifiers[classifier](train_x, train_y)
    print('training took %fs!' % (time.time() - start_time))
    result_buffer += ('training took %fs!' % (time.time() - start_time) + '\n')
    predict = model.predict(test_x)

    if is_binary_class:  # 二分类才去计算PR
    # if True:
        precision = metrics.precision_score(test_y, predict)
        recall = metrics.recall_score(test_y, predict)
        print('precision: %.2f%%, \nrecall: %.2f%%' % (100 * precision, 100 * recall))
        result_buffer += ('precision: %.2f%%, \nrecall: %.2f%%' % (100 * precision, 100 * recall) + '\n')
    accuracy = metrics.accuracy_score(test_y, predict)
    print('accuracy: %.2f%%' % (100 * accuracy))
    result_buffer += ('accuracy: %.2f%%' % (100 * accuracy) + '\n')
    print("\n")
    result_buffer += '\n'

with open(save_data_path_selected_data_path + "/log.txt", 'a+') as f:
    f.write(result_buffer)
