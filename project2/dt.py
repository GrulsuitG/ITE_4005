import itertools
import math
import random

import numpy as np
import pandas as pd
import sys


# Information Gain / Gain Ratio measure 계산
def entropy(target):
    _, count = np.unique(target, return_counts=True)
    count = count / sum(count)
    return sum(-p * math.log2(p) for p in count)


# Gini Index measure 계산
def get_gini(target):
    _, count = np.unique(target, return_counts=True)
    count = count / sum(count)

    return 1 - sum(p * p for p in count)


# InformationGain 을 구하는 함수
def information_gain(dataset, split_attribute, target_attribute):
    before_info = entropy(dataset[target_attribute])

    value, count = np.unique(dataset[split_attribute], return_counts=True)
    after_info = 0
    total = sum(count)
    for val, cnt in zip(value, count):
        child = dataset[dataset[split_attribute] == val]
        after_info += (cnt / total) * entropy(child[target_attribute])

    return before_info - after_info


# Gain Ratio 를 구하는 함수
def gain_ratio(dataset, split_attribute, target_attribute):
    info = information_gain(dataset, split_attribute, target_attribute)
    ratio = entropy(dataset[split_attribute])
    if ratio == 0:
        return 0
    return info / ratio


# Gini Index 를 구하는 함수
# Best Combination 도 return 한다
def gini_index(dataset, split_attribute, target_attribute):
    value = set(np.unique(dataset[split_attribute]))
    total = len(dataset)

    # 현재 level 의 gini 계산
    before_gini = get_gini(dataset[target_attribute])

    best_split = []
    diff_gini = 0
    # value 의 갯수에 따라 여러 가지 경우의 수를 다 split 해본다.
    # ex) value 가 4개 -> 1/3, 2/2
    for i in range(1, (len(value) // 2) + 1):
        combinations = map(set, itertools.combinations(value, i))
        for comb in combinations:
            counter = value - comb
            child_a = dataset[dataset[split_attribute].isin(comb)]
            child_b = dataset[dataset[split_attribute].isin(counter)]
            child_gini = (len(child_a) / total) * get_gini(child_a[target_attribute]) + \
                         (len(child_b) / total) * get_gini(child_b[target_attribute])

            result = before_gini - child_gini
            if result >= diff_gini:
                diff_gini = result
                best_split = [tuple(comb), tuple(counter)]

    return best_split, diff_gini


# 정해진 split 에 따라 child 를 구하는 함수
def split_data(dataset, split, comb=None):
    result = []

    # Information Gain / Gain Ratio
    if comb is None:
        value = np.unique(dataset[split])
        for val in value:
            result.append(dataset[dataset[split] == val])
        return result, value

    # Gini index
    else:
        for c in comb:
            result.append(dataset[dataset[split].isin(c)])

    # split attribute 을 구할 때 combination 을 구해서 value 를 return 할 필요 없다
    return result


# decision tree 를 만드는 함수
def make_decision_tree(dataset, split_attributes, target_attribute):
    classification, count = np.unique(dataset[target_attribute], return_counts=True)

    # target attribute 가 1개로 나와서 모두 분류된 경우
    if len(classification) == 1:
        # print('classify clear')
        return classification[0]

    major = find_major(classification, count)
    minor = find_minor(classification, count)

    # 만약 더이상 분기할 attribute 가 없으면 target_attribute 중 가장 major 한 값을 return 한다
    # 일종의 예외 처리
    if len(split_attributes) == 0:
        # print('no split attributes')
        return major
        # return minor

    split = split_attributes[0]
    gain = 0
    best_comb = []
    for attr in split_attributes:
        # info = information_gain(dataset, attr, target_attribute)
        # info = gain_ratio(dataset, attr, target_attribute)
        comb, info = gini_index(dataset, attr, target_attribute)
        if gain <= info:
            gain = info
            split = attr
            best_comb = comb

    # Information Gain / Gain Ratio
    # split_attributes = split_attributes.drop(split)
    # children, value = split_data(dataset, split)
    #
    # tree = {}
    # for child, val in zip(children, value):
    #     tree[val] = make_decision_tree(child, split_attributes, target_attribute)
    # # tree[None] = major
    # # tree[None] = minor
    # tree[None] = make_random_value(dataset[target_attribute])
    # return {split: tree}
    # Information Gain / Gain Ratio End

    # Gini Index
    children = split_data(dataset, split, best_comb)
    tree = {}
    for child, comb in zip(children, best_comb):
        next_split_attributes = split_attributes
        # attribute 의 value 가 무조건 binary 형태, 따라서 현재 나눈 attribute 가 child tree 에도 있을 수 있다
        if len(comb) == 1:
            next_split_attributes = next_split_attributes.drop(split)

        tree[comb] = make_decision_tree(child, next_split_attributes, target_attribute)
        tree[None] = major
        # tree[None] = minor

    return {split: tree}
    # Gini Index End


# 현재 단계의 target 중 가장 많은 값을 찾는 함수
def find_major(classification, count):
    major = 0
    result = classification[0]
    for cnt, target in zip(count, classification):
        if major <= cnt:
            result = target
            major = cnt
    return result


# 현재 단계의 target 중 가장 적은 값을 찾는 함수
def find_minor(classification, count):
    minor = math.inf
    result = classification[0]
    for cnt, target in zip(count, classification):
        if minor >= cnt:
            result = target
            minor = cnt
    return result


def make_random_value(classification):
    target = np.unique(classification)
    idx = random.randint(0, len(target) - 1)
    print(target, idx)
    return target[idx]


# Information Gain / Gain Ratio Tree 로 classify 하는 함수
def classify_by_info(item, tree):
    if type(tree) != dict:
        return tree

    attr = list(tree.keys())[0]
    child = tree[attr]
    feature = item[attr]

    if child.get(feature) is None:
        subtree = child[None]
    else:
        subtree = child[feature]

    return classify_by_info(item, subtree)


# Gini Index Tree 로 classify 하는 함수
# tree 의 key 가 tuple 이기 때문에 별도로 찾는 과정 필요
def classify_by_gini(item, tree):
    if type(tree) != dict:
        return tree

    attr = list(tree.keys())[0]
    child = tree[attr]
    feature = item[attr]

    for key in child.keys():
        if key is not None:
            if feature in key:
                feature = key
                break
    if child.get(feature) is None:
        subtree = child[None]
    else:
        subtree = child[feature]

    return classify_by_gini(item, subtree)


def main():
    argv = sys.argv

    # 디버깅 용
    training_file = 'dt_train1.txt'
    test_file = 'dt_test1.txt'
    output_file = 'test/dt_result1.txt'

    if len(argv) != 1:
        training_file = argv[1]
        test_file = argv[2]
        output_file = argv[3]

    training = pd.read_csv(training_file, sep='\t')
    test = pd.read_csv(test_file, sep='\t')

    target_attribute = training.columns[-1]
    tree = make_decision_tree(training, training.columns[:-1], target_attribute)

    # classification = [classify_by_info(test.iloc[idx], tree) for idx in range(len(test))]
    classification = [classify_by_gini(test.iloc[idx], tree) for idx in range(len(test))]

    result = pd.DataFrame(test)
    result[target_attribute] = classification

    result.to_csv(output_file, sep='\t', index=False)


if __name__ == '__main__':
    main()
