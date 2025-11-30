import os
import sys
import argparse
import logging
import math
import random

# ====== Feature Engineering & Hyperparameter Candidates ======
# 최소 2개 이상의 feature set을 정의해서 자동으로 비교
FEATURE_SETS = {
    # 모든 센서/피처 사용 (기본)
    "full": [1, 2, 3, 4, 5, 6, 7],
    # 비교용: 앞부분 피처만 사용
    "env_only": [1, 2, 3, 4],
    # 비교용: 뒤쪽(이벤트/부하 관련) 피처만 사용
    "event_only": [3, 4, 5, 6, 7],
}

# 분산에 더해 줄 smoothing 값 후보 (하이퍼파라미터)
VAR_SMOOTHING_CANDIDATES = [1e-6, 1e-4, 1e-2]


# ====== Core Naive Bayes Training (single config) ======
def _fit_gaussian_nb(instances, labels, feature_indices, var_smoothing):
    """
    주어진 feature_indices와 var_smoothing(분산 smoothing) 설정으로
    Gaussian Naive Bayes 파라미터를 학습하는 함수.
    """
    n = len(instances)
    # if n == 0:
    #     logging.error("No training data.")
    #     sys.exit(1)

    # 클래스별 prior P(y)
    class_counts = {}
    for y in labels:
        class_counts[y] = class_counts.get(y, 0) + 1

    priors = {}
    for y, cnt in class_counts.items():
        priors[y] = cnt / n

    # 클래스별, 특성별 평균/분산 (Gaussian Naive Bayes)
    means = {y: {i: 0.0 for i in feature_indices} for y in class_counts.keys()}
    vars_ = {y: {i: 0.0 for i in feature_indices} for y in class_counts.keys()}

    # 평균 계산
    for x, y in zip(instances, labels):
        for i in feature_indices:
            means[y][i] += float(x[i])

    for y in class_counts.keys():
        for i in feature_indices:
            means[y][i] /= class_counts[y]

    # 분산 계산
    for x, y in zip(instances, labels):
        for i in feature_indices:
            diff = float(x[i]) - means[y][i]
            vars_[y][i] += diff * diff

    for y in class_counts.keys():
        for i in feature_indices:
            if class_counts[y] > 1:
                vars_[y][i] /= class_counts[y] - 1
            else:
                vars_[y][i] = 0.0

            # 분산이 0이 되지 않도록 smoothing 추가 (하이퍼파라미터)
            vars_[y][i] += var_smoothing
            if vars_[y][i] == 0.0:
                vars_[y][i] = var_smoothing

    parameters = {
        "priors": priors,
        "means": means,
        "vars": vars_,
        "feature_indices": feature_indices,
        "classes": list(class_counts.keys()),
        "var_smoothing": var_smoothing,
    }

    return parameters


# ====== Metric 계산 유틸 ======
def compute_metrics(predictions, answers):
    # if len(predictions) != len(answers):
    #     logging.error("The lengths of two arguments should be same")
    #     sys.exit(1)

    # accuracy
    correct = 0
    for idx in range(len(predictions)):
        if predictions[idx] == answers[idx]:
            correct += 1
    accuracy = correct / len(answers) if answers else 0.0

    # precision (positive class = 1)
    tp = 0
    fp = 0
    for idx in range(len(predictions)):
        if predictions[idx] == 1:
            if answers[idx] == 1:
                tp += 1
            else:
                fp += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # recall (positive class = 1)
    tp = 0
    fn = 0
    for idx in range(len(answers)):
        if answers[idx] == 1:
            if predictions[idx] == 1:
                tp += 1
            else:
                fn += 1
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


# ====== K-fold evaluation for parameter tuning ======
def _evaluate_config_kfold(instances, labels, feature_indices, var_smoothing, k=5):
    """
    하나의 (feature set, var_smoothing) 설정에 대해
    k-fold cross-validation으로 성능(accuracy)을 평가.
    """
    n = len(instances)
    if n < 2:
        return 0.0

    indices = list(range(n))
    random.shuffle(indices)

    k = min(k, n)
    fold_size = n // k
    if fold_size == 0:
        # 데이터가 매우 적으면 hold-out 방식
        train_idx = indices[:-1]
        test_idx = indices[-1:]
        train_instances = [instances[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_instances = [instances[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        params = _fit_gaussian_nb(
            train_instances, train_labels, feature_indices, var_smoothing
        )
        preds = [predict(x, params) for x in test_instances]
        metric = compute_metrics(preds, test_labels)
        return metric["accuracy"]

    acc_sum = 0.0
    for fold in range(k):
        start = fold * fold_size
        end = n if fold == k - 1 else (fold + 1) * fold_size
        test_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]

        train_instances = [instances[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_instances = [instances[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        params = _fit_gaussian_nb(
            train_instances, train_labels, feature_indices, var_smoothing
        )
        preds = [predict(x, params) for x in test_instances]
        metric = compute_metrics(preds, test_labels)
        acc_sum += metric["accuracy"]

    return acc_sum / k


def training(instances, labels):
    pass


def predict(instance, parameters):
    pass


def report(predictions, answers):
    if len(predictions) != len(answers):
        logging.error("The lengths of two arguments should be same")
        sys.exit(1)

    # accuracy
    correct = 0
    for idx in range(len(predictions)):
        if predictions[idx] == answers[idx]:
            correct += 1
    accuracy = round(correct / len(answers), 2) * 100

    # precision
    tp = 0
    fp = 0
    for idx in range(len(predictions)):
        if predictions[idx] == 1:
            if answers[idx] == 1:
                tp += 1
            else:
                fp += 1
    precision = round(tp / (tp + fp), 2) * 100

    # recall
    tp = 0
    fn = 0
    for idx in range(len(answers)):
        if answers[idx] == 1:
            if predictions[idx] == 1:
                tp += 1
            else:
                fn += 1
    recall = round(tp / (tp + fn), 2) * 100

    logging.info("accuracy: {}%".format(accuracy))
    logging.info("precision: {}%".format(precision))
    logging.info("recall: {}%".format(recall))


def load_raw_data(fname):
    instances = []
    labels = []
    with open(fname, "r") as f:
        f.readline()
        for line in f:
            tmp = line.strip().split(", ")
            tmp[1] = float(tmp[1])
            tmp[2] = float(tmp[2])
            tmp[3] = float(tmp[3])
            tmp[4] = float(tmp[4])
            tmp[5] = int(tmp[5])
            tmp[6] = int(tmp[6])
            tmp[7] = float(tmp[7])
            tmp[8] = int(tmp[8])
            instances.append(tmp[:-1])
            labels.append(tmp[-1])
    return instances, labels


def run(train_file, test_file):
    # training phase
    instances, labels = load_raw_data(train_file)
    logging.debug("instances: {}".format(instances))
    logging.debug("labels: {}".format(labels))
    parameters = training(instances, labels)

    # testing phase
    instances, labels = load_raw_data(test_file)
    predictions = []
    for instance in instances:
        result = predict(instance, parameters)

        if result not in [0, 1]:
            logging.error("The result must be either 0 or 1")
            sys.exit(1)

        predictions.append(result)

    # report
    report(predictions, labels)


def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--training",
        required=True,
        metavar="<file path to the training dataset>",
        help="File path of the training dataset",
        default="training.csv",
    )
    parser.add_argument(
        "-u",
        "--testing",
        required=True,
        metavar="<file path to the testing dataset>",
        help="File path of the testing dataset",
        default="testing.csv",
    )
    parser.add_argument(
        "-l",
        "--log",
        help="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)",
        type=str,
        default="INFO",
    )

    args = parser.parse_args()
    return args


def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    if not os.path.exists(args.training):
        logging.error("The training dataset does not exist: {}".format(args.training))
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error("The testing dataset does not exist: {}".format(args.testing))
        sys.exit(1)

    run(args.training, args.testing)


if __name__ == "__main__":
    main()
