from itertools import combinations
import sys

transaction = []
minimum_support = 5.0 / 100
minimum_cnt = 5

def main():
    global minimum_cnt, minimum_support, transaction
    argv = sys.argv
    minimum_support = float(argv[1]) / 100

    input_file_name = argv[2]
    # input_file_name = 'input.txt'

    output_file_name = argv[3]
    # output_file_name = 'output.txt'

    # 파일을 읽고 각각의 transaction을 list 형태로 저장
    with open(input_file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            transaction.append(line.split())

    minimum_cnt = len(transaction) * minimum_support
    # apriori 알고리즘 실행
    result = apriori()

    # 결과를 output file에 저장
    with open(output_file_name, 'w') as f:
        f.write(result)


# apriori 알고리즘 메인 함수
def apriori():
    # 길이가 1인 초기 frequnt item set 을 세팅
    frequent_item = [init_frequent_item_set()]

    length = 1
    while True:
        prev_frequent_item = frequent_item[length - 1]

        candidate = make_candidate(prev_frequent_item, length + 1)
        frequent = scan_transaction(candidate)
        if len(frequent) == 0:
            break
        else:
            frequent_item.append(frequent)
            length += 1

    # 찾은 frequent item 을 바탕으로 association rule을 찾는다.
    return association_rule(frequent_item)


# 길이가 1인 frequent item 을 세팅
def init_frequent_item_set():
    item_set = {}

    for transact in transaction:
        for item in transact:
            if item in item_set:
                item_set[item] += 1
            else:
                item_set[item] = 1

    return check_minimum_support(item_set)


# item_set이 입력받은 minimum support 를 넘는지 확인
def check_minimum_support(item_set):
    frequent_item = {}
    for key, value in item_set.items():
        if value >= minimum_cnt:
            frequent_item[key] = value
    return frequent_item


# 다음 후보를 찾는 함수
def make_candidate(frequent_item, length):
    if length == 2:
        return list(combinations(frequent_item, length))
    else:
        element = []
        for items in frequent_item:
            for item in items:
                if item not in element:
                    element.append(item)
        temp_candidate = list(combinations(element, length))
        candidate = []

        set_list = [set(item) for item in frequent_item]
        # Downward Closure
        for items in temp_candidate:
            is_candidate = True
            for subset in list(combinations(items, length - 1)):
                subset = set(subset)
                if subset not in set_list:
                    is_candidate = False
                    break

            # dictionary 키로 사용하기 위해 tuple 형태로 저장
            if is_candidate:
                candidate.append(tuple(items))

        return candidate


# 찾은 후보를 대상으로 transaction을 스캔해 frequent item 인지 확인
def scan_transaction(candidate):
    item_set = {}

    for transact in transaction:
        for item in candidate:
            if set(transact) >= set(item):
                if item_set.get(item) is None:
                    item_set[item] = 1
                else:
                    item_set[item] += 1

    return check_minimum_support(item_set)


# confidence를 찾기위해 frequent item 중 후보의 count를 찾는 부분
def find_confidence(candidate, frequent_item):
    if len(candidate) == 1:
        candidate = candidate[0]
    for items in frequent_item:
        for item in items:
            if set(item) == set(candidate):
                return items.get(item)

# 찾은 frequent item 을 대상으로 association rule을 찾는다.
def association_rule(frequent_item):
    item_length = 1
    transaction_length = len(transaction)
    result = ''
    for item in frequent_item:
        for key, value in item.items():
            length = item_length

            while length > 1:

                candidates = list(combinations(key, length - 1))
                for candidate in candidates:
                    '''
                     candidate(A) 와 association(B) 의 support 는  A의 value or B의 value ( == A+B 의 value)
                     A->B confidence 는 A+B value / A  value 
                    '''

                    association = set(map(int, (set(key) - set(candidate))))

                    support = float(value) / transaction_length * 100
                    support = round(support, 2)

                    confidence = float(value) / find_confidence(candidate, frequent_item) * 100
                    confidence = round(confidence, 2)

                    candidate = set(map(int, candidate))

                    result += "{}\t{}\t{:.2f}\t{:.2f}\n".format(str(candidate), str(association), support, confidence)

                length -= 1
        item_length += 1

    return result


if __name__ == '__main__':
    main()
