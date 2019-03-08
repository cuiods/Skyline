import pandas
import numpy as np
import scipy
import statsmodels.api as sm
import traceback
import logging
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
)

from algorithm_exceptions import *

logger = logging.getLogger("AnalyzerLog")
redis_conn = StrictRedis(unix_socket_path=REDIS_SOCKET_PATH)
centers = np.zeros((1, 1))
avg_score = -1

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
    if centers.shape[0] <= 1:
        update_vertex_param()
    test_data = timeseries[:, 1:]
    test_data = (test_data - np.min(test_data, axis=0)) / (np.max(test_data, axis=0) - np.min(test_data, axis=0))
    test_data = np.nan_to_num(test_data)
    score = calculate_vertex_score(test_data, centers)
    if np.sum(score[score > avg_score]) > VERTEX_THRESHOLD:
        return True
    return False


def update_vertex_param():
    """
    Read observed abnormal data and update cluster centers
    """
    global centers
    global avg_score
    origin_data = pandas.read_csv("data/data_ipv6_abnormal")
    abnormal = origin_data[:, 3:]
    abnormal = (abnormal - np.min(abnormal, axis=0)) / (np.max(abnormal, axis=0) - np.min(abnormal, axis=0))
    abnormal = np.nan_to_num(abnormal)
    k_means = KMeans(n_clusters=K_MEANS_CLUSTER)
    k_means.fit_predict(abnormal)
    centers = k_means.cluster_centers_
    avg_score = np.mean(calculate_vertex_score(abnormal, centers))


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
    try:
        t = (timeseries[-1][1] + timeseries[-2][1] + timeseries[-3][1]) / 3
        return t
    except IndexError:
        return timeseries[-1][1]


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
    last_hour_threshold = time() - (FULL_DURATION - 3600)
    series = pandas.Series([x[1] for x in timeseries if x[0] < last_hour_threshold])
    mean = (series).mean()
    stdDev = (series).std()
    t = tail_avg(timeseries)

    return abs(t - mean) > 3 * stdDev


def stddev_from_average(timeseries):
    """
    A timeseries is anomalous if the absolute value of the average of the latest
    three datapoint minus the moving average is greater than three standard
    deviations of the average. This does not exponentially weight the MA and so
    is better for detecting anomalies with respect to the entire series.
    """
    series = pandas.Series([x[1] for x in timeseries])
    mean = series.mean()
    stdDev = series.std()
    t = tail_avg(timeseries)

    return abs(t - mean) > 3 * stdDev


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
