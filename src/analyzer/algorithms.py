# coding=utf-8
import pandas
import numpy as np
import scipy
import statsmodels.api as sm
import traceback
import logging
import math
from time import time
from msgpack import unpackb, packb
from redis import StrictRedis
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

from settings import (
    ALGORITHMS,
    CONSENSUS,
    FULL_DURATION,
    MAX_TOLERABLE_BOREDOM,
    MIN_TOLERABLE_LENGTH,
    STALE_PERIOD,
    REDIS_SOCKET_PATH,
    ENABLE_SECOND_ORDER,
    BOREDOM_SET_SIZE,
    K_MEANS_CLUSTER,
    VERTEX_WEIGHT_ETA,
    VERTEX_THRESHOLD,
    ANOMALY_COLUMN,
    ANOMALY_PATH,
)

from algorithm_exceptions import *

logger = logging.getLogger("AnalyzerLog")
redis_conn = StrictRedis(unix_socket_path=REDIS_SOCKET_PATH)
vertex_centers = np.zeros((1, 1))
vertex_avg_score = -1
cshl_weight = np.zeros((1, 1))

"""
This is no man's land. Do anything you want in here,
as long as you return a boolean that determines whether the input
timeseries is anomalous or not.

To add an algorithm, define it here, and add its name to settings.ALGORITHMS.
"""


def vertex_score(timeseries):
    """
    A timeseries is anomalous if vertex score in hypergraph is greater than average score of observed anomalous vertex.
    :return: True or False
    """
    if vertex_centers.shape[0] <= 1:
        update_vertex_param()
    timeseries = np.array(timeseries)
    test_data = timeseries[:, 1:]
    test_data = (test_data - np.min(test_data, axis=0)) / (np.max(test_data, axis=0) - np.min(test_data, axis=0))
    test_data = np.nan_to_num(test_data)
    score = calculate_vertex_score(test_data, vertex_centers)
    if np.sum(score[score > vertex_avg_score]) > VERTEX_THRESHOLD:
        return True
    return False


def update_vertex_param():
    """
    Read observed abnormal data and update cluster centers
    """
    global vertex_centers
    global vertex_avg_score
    origin_data = pandas.read_csv(ANOMALY_PATH).values
    abnormal = origin_data[:, 3:]
    abnormal = (abnormal - np.min(abnormal, axis=0)) / (np.max(abnormal, axis=0) - np.min(abnormal, axis=0))
    abnormal = np.nan_to_num(abnormal)
    k_means = KMeans(n_clusters=K_MEANS_CLUSTER)
    k_means.fit_predict(abnormal)
    vertex_centers = k_means.cluster_centers_
    vertex_avg_score = np.mean(calculate_vertex_score(abnormal, vertex_centers))


def calculate_vertex_score(samples, center):
    """
    we use similarity score and isolation score to initialize vertex weight
    according to their correlations

    :param samples: all the samples
    :param center: abnormal cluster center
    :return: total score of samples
    """
    clf = IsolationForest()
    clf.fit(samples)
    num = samples.shape[0]
    IS = (0.5 - clf.decision_function(samples)).reshape((num, 1))
    distance = np.array(np.min(euclidean_distances(samples, center), axis=1))
    dis_min = np.min(distance)
    dis_max = np.max(distance)
    distance = (distance - dis_min) / (dis_max - dis_min)
    SS = np.exp(-distance).reshape((num, 1))
    TS = VERTEX_WEIGHT_ETA * IS + (1-VERTEX_WEIGHT_ETA) * SS
    return TS


def euclidean_distances(A, B):
    """
    Euclidean distance between matrix A and B

    :param A: np array
    :param B: np array
    :return: np array
    """
    BT = B.transpose()
    vec_prod = np.dot(A, BT)
    SqA =  A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vec_prod.shape[1]))
    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vec_prod.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vec_prod
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


def tail_avg(timeseries):
    """
    This is a utility function used to calculate the average of the last three
    datapoints in the series as a measure, instead of just the last datapoint.
    It reduces noise, but it also reduces sensitivity and increases the delay
    to detection.
    """
    timeseries = np.array(timeseries)
    timeseries = timeseries[:, 1:]
    try:
        t = (timeseries[-1] + timeseries[-2] + timeseries[-3]) / 3
        return t
    except IndexError:
        return timeseries[-1]


def z_score(x):
    """
    z-standard
    :param x:
    :return:
    """
    dim = x.shape[1]
    leng = x.shape[0]
    for i in range(dim):
        total = 0
        var = 0
        for j in range(leng):
            total += x[j][i]
        ave = float(total) / leng
        for j in range(leng):
            var += pow(x[j][i] - ave, 2)
        var = var / (leng - 1)
        var = pow(var, 0.5)
        for j in range(leng):
            x[j][i] = (x[j][i] - ave) / var
    return x


def hpconstruct(x, y, k):
    """
    construct hypergraph and interative process
    :param x: np array, train and test set
    :param y: np array, cost for each sample
    :param k: value, kNN
    :return: evaluation criteria
    """
    length = len(x)
    h = np.zeros((length, length))
    dvlist = []
    delist = []
    totaldis = 0.0
    alpha = 0.05

    wm = np.eye(length)
    wm = (1.0 / length) * wm
    # initialize W

    for xi in range(length):
        diffMat = np.tile(x[xi], (length, 1)) - x  # 求inX与训练集各个实例的差
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5  # 求欧式距离
        sortedDistIndicies = distances.argsort()  # 取排序的索引，用于排label
        for i in range(k):
            index = sortedDistIndicies[i + 1]
            h[index][xi] = distances[index]
        totaldis += distances.sum()
    avedis = totaldis / (length ** 2 - length)
    for xi in range(length):
        for yi in range(length):
            if h[xi][yi]:
                h[xi][yi] = math.exp(((h[xi][yi] / avedis) ** 2) / (-alpha))
        h[xi][xi] = 1
    # initialize H，横坐标代表点，纵坐标代表边（中心点为序号）

    for xi in range(length):
        vertextmp = 0
        for yi in range(length):
            vertextmp += wm[yi][yi] * h[xi][yi]
        dvlist.append(vertextmp)
    dv = np.diag(dvlist)
    # initialize Dv

    for xi in range(length):
        edgetmp = 0
        for yi in range(length):
            edgetmp += h[yi][xi]
        delist.append(edgetmp)
    de = np.diag(delist)
    # initialize De

    di = []
    # y = np.array([])
    for i in range(length):
        if y[i] == 1:
            di.append(1)
        elif y[i] == -1:
            di.append(1)
        else:
            di.append(0)
    v = np.diag(di)
    # initialize Υ

    for i in range(length):
        dv[i][i] = 1 / (math.sqrt(dv[i][i]))
        # de[i][i] = 1 / de[i][i]
    # calculate power of Dv and De
    de = np.linalg.inv(de)
    mu = 1
    lamb = 1

    xt = np.transpose(x)
    first = np.dot(v, v)
    first = np.dot(xt, first)
    first = np.dot(first, x)
    third = np.dot(xt, v)
    third = np.dot(third, y)
    second = mu * xt
    # initialize fixed part of ω

    count = 0
    threshold = 0.000001
    opt = [0]

    while True:
        deltaleft = np.dot(dv, h)
        deltaright = np.dot(de, np.transpose(h))
        deltaright = np.dot(deltaright, dv)
        deltai = np.eye(length)
        # left and right part of Δ

        delta = np.dot(deltaleft, wm)
        delta = np.dot(delta, deltaright)
        delta = deltai - delta
        # first delta

        w = np.dot(second, delta)
        w = np.dot(w, x)
        w = first + w
        w = np.linalg.inv(w)
        w = np.dot(w, third)
        # first w

        xw = np.dot(x, w)
        tmp = xw - y
        tmp = np.dot(v, tmp)
        remp = np.linalg.norm(tmp, ord=2) ** 2
        omega = np.dot(np.transpose(xw), delta)
        omega = np.dot(omega, xw)
        kesai = np.linalg.norm(wm) ** 2
        opttmp = remp + mu * omega + lamb * kesai
        opt.append(opttmp)
        # first optimization

        count += 1
        if count > 2 and opt[count - 1] - opt[count] < threshold:
            break
        # judge

        caplambda = np.dot(np.transpose(xw), dv)
        caplambda = np.dot(caplambda, h)
        # first Λ

        yita_tmp = mu * caplambda
        yita_tmp = np.dot(yita_tmp, de)
        yita_tmp = np.dot(yita_tmp, np.transpose(caplambda))
        yita = (yita_tmp - 2 * lamb) / length
        # first η

        w_tmp = mu * np.transpose(caplambda)
        w_tmp = np.dot(w_tmp, caplambda)
        wm = (0.5 / lamb) * (np.dot(w_tmp, de) - yita * deltai)
        # second W

        dvlist = []
        for xi in range(length):
            vertextmp = 0
            for yi in range(length):
                vertextmp += wm[yi][yi] * h[xi][yi]
            dvlist.append(vertextmp)
        dv = np.diag(dvlist)
        for i in range(length):
            dv[i][i] = 1 / (math.sqrt(dv[i][i]))
        # initialize Dv
    # iteration

    return w


def grubbs(timeseries):
    """
    A timeseries is anomalous if the Z score is greater than the Grubb's score.
    """

    series = scipy.array([x[1] for x in timeseries])
    stdDev = scipy.std(series)
    mean = np.mean(series)
    tail_average = tail_avg(timeseries)
    z_score = (tail_average - mean) / stdDev
    len_series = len(series)
    threshold = scipy.stats.t.isf(.05 / (2 * len_series), len_series - 2)
    threshold_squared = threshold * threshold
    grubbs_score = ((len_series - 1) / np.sqrt(len_series)) * np.sqrt(threshold_squared / (len_series - 2 + threshold_squared))

    return z_score > grubbs_score


def first_hour_average(timeseries):
    """
    Calcuate the simple average over one hour, FULL_DURATION seconds ago.
    A timeseries is anomalous if the average of the last three datapoints
    are outside of three standard deviations of this value.
    """
    timeseries = np.array(timeseries)
    last_hour_threshold = time() - (FULL_DURATION - 3600)
    series = timeseries[timeseries[:, 0] < last_hour_threshold]
    if series.shape[0] == 0:
        return False
    series = series[:, 1:]
    mean = np.mean(series, axis=0)
    stdDev = np.std(series, axis=0)
    t = tail_avg(timeseries)

    return np.sum(abs(t - mean) > 3 * stdDev) > ANOMALY_COLUMN


def stddev_from_average(timeseries):
    """
    A timeseries is anomalous if the absolute value of the average of the latest
    three datapoint minus the moving average is greater than three standard
    deviations of the average. This does not exponentially weight the MA and so
    is better for detecting anomalies with respect to the entire series.
    """
    timeseries = np.array(timeseries)
    series = timeseries[:, 1]
    mean = np.mean(series, axis=0)
    stdDev = np.std(series, axis=0)
    t = tail_avg(timeseries)

    return np.sum(abs(t - mean) > 3 * stdDev) > ANOMALY_COLUMN


def least_squares(timeseries):
    """
    A timeseries is anomalous if the average of the last three datapoints
    on a projected least squares model is greater than three sigma.
    """

    x = np.array([t[0] for t in timeseries])
    y = np.array([t[1] for t in timeseries])
    A = np.vstack([x, np.ones(len(x))]).T
    results = np.linalg.lstsq(A, y)
    residual = results[1]
    m, c = np.linalg.lstsq(A, y)[0]
    errors = []
    for i, value in enumerate(y):
        projected = m * x[i] + c
        error = value - projected
        errors.append(error)

    if len(errors) < 3:
        return False

    std_dev = scipy.std(errors)
    t = (errors[-1] + errors[-2] + errors[-3]) / 3

    return abs(t) > std_dev * 3 and round(std_dev) != 0 and round(t) != 0


def histogram_bins(timeseries):
    """
    A timeseries is anomalous if the average of the last three datapoints falls
    into a histogram bin with less than 20 other datapoints (you'll need to tweak
    that number depending on your data)

    Returns: the size of the bin which contains the tail_avg. Smaller bin size
    means more anomalous.
    """

    series = scipy.array([x[1] for x in timeseries])
    t = tail_avg(timeseries)
    h = np.histogram(series, bins=15)
    bins = h[1]
    for index, bin_size in enumerate(h[0]):
        if bin_size <= 20:
            # Is it in the first bin?
            if index == 0:
                if t <= bins[0]:
                    return True
            # Is it in the current bin?
            elif t >= bins[index] and t < bins[index + 1]:
                    return True

    return False


def ks_test(timeseries):
    """
    A timeseries is anomalous if 2 sample Kolmogorov-Smirnov test indicates
    that data distribution for last 10 minutes is different from last hour.
    It produces false positives on non-stationary series so Augmented
    Dickey-Fuller test applied to check for stationarity.
    """

    hour_ago = time() - 3600
    ten_minutes_ago = time() - 600
    reference = scipy.array([x[1] for x in timeseries if x[0] >= hour_ago and x[0] < ten_minutes_ago])
    probe = scipy.array([x[1] for x in timeseries if x[0] >= ten_minutes_ago])

    if reference.size < 20 or probe.size < 20:
        return False

    ks_d, ks_p_value = scipy.stats.ks_2samp(reference, probe)

    if ks_p_value < 0.05 and ks_d > 0.5:
        adf = sm.tsa.stattools.adfuller(reference, 10)
        if adf[1] < 0.05:
            return True

    return False


def is_anomalously_anomalous(metric_name, ensemble, datapoint):
    """
    This method runs a meta-analysis on the metric to determine whether the
    metric has a past history of triggering. TODO: weight intervals based on datapoint
    """
    # We want the datapoint to avoid triggering twice on the same data
    new_trigger = [time(), datapoint]

    # Get the old history
    raw_trigger_history = redis_conn.get('trigger_history.' + metric_name)
    if not raw_trigger_history:
        redis_conn.set('trigger_history.' + metric_name, packb([(time(), datapoint)]))
        return True

    trigger_history = unpackb(raw_trigger_history)

    # Are we (probably) triggering on the same data?
    if (new_trigger[1] == trigger_history[-1][1] and
            new_trigger[0] - trigger_history[-1][0] <= 300):
                return False

    # Update the history
    trigger_history.append(new_trigger)
    redis_conn.set('trigger_history.' + metric_name, packb(trigger_history))

    # Should we surface the anomaly?
    trigger_times = [x[0] for x in trigger_history]
    intervals = [
        trigger_times[i + 1] - trigger_times[i]
        for i, v in enumerate(trigger_times)
        if (i + 1) < len(trigger_times)
    ]

    series = pandas.Series(intervals)
    mean = series.mean()
    stdDev = series.std()

    return abs(intervals[-1] - mean) > 3 * stdDev


def run_selected_algorithm(timeseries, metric_name):
    """
    Filter timeseries and run selected algorithm.
    """
    # Get rid of short series
    if len(timeseries) < MIN_TOLERABLE_LENGTH:
        raise TooShort()

    # Get rid of stale series
    if time() - timeseries[-1][0] > STALE_PERIOD:
        raise Stale()

    # Get rid of boring series
    if len(set(item[1] for item in timeseries[-MAX_TOLERABLE_BOREDOM:])) == BOREDOM_SET_SIZE:
        raise Boring()

    try:
        ensemble = [globals()[algorithm](timeseries) for algorithm in ALGORITHMS]
        threshold = len(ensemble) - CONSENSUS
        if ensemble.count(False) <= threshold:
            # if ENABLE_SECOND_ORDER:
            #     if is_anomalously_anomalous(metric_name, ensemble, timeseries[-1]):
            #         return True, ensemble, timeseries[-1]
            # else:
            #     return True, ensemble, timeseries[-1]
            return True, ensemble, timeseries[-1]
        return False, ensemble, timeseries[-1]
    except:
        logging.error("Algorithm error: " + traceback.format_exc())
        return False, [], 1
