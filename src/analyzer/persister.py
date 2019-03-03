from threading import Thread
from redis import StrictRedis
from os import kill
from time import sleep, time
from multiprocessing import Process
from math import ceil
from msgpack import Unpacker
import pymysql

import logging
import settings

logger = logging.getLogger("AnalyzerLog")


class Persister(Thread):
    """
    The Persister is responsible for saving anomaly data points to mysql database.
    """
    def __init__(self, parent_pid):
        super(Persister, self).__init__()
        self.redis_conn = StrictRedis(unix_socket_path=settings.REDIS_SOCKET_PATH)
        self.mysql_conn = pymysql.connect(host=settings.MYSQL_HOST, port=settings.MYSQL_PORT, user=settings.MYSQL_USER,
                                          password=settings.MYSQL_PASSWORD, db=settings.MYSQL_DB, charset=settings.MYSQL_CHARSET)
        self.parent_pid = parent_pid

    def check_if_parent_is_alive(self):
        """
        Self explanatory.
        """
        try:
            kill(self.parent_pid, 0)
        except:
            exit(0)

    def do_persist(self, index, anomaly_unique_metrics):
        """
        Assign a bunch of anomaly metrics for a process to persist.
        """
        # Discover assigned metrics
        keys_per_processor = int(ceil(float(len(anomaly_unique_metrics)) / float(settings.PERSIST_PROCESSES)))
        if index == settings.PERSIST_PROCESSES:
            assigned_max = len(anomaly_unique_metrics)
        else:
            assigned_max = index * keys_per_processor
        assigned_min = assigned_max - keys_per_processor
        assigned_keys = range(assigned_min, assigned_max)

        # Compile assigned metrics
        assigned_metrics = [anomaly_unique_metrics[index] for index in assigned_keys]

        # Check if this process is unnecessary
        if len(assigned_metrics) == 0:
            return

        # Multi get series
        pipe = self.redis_conn.pipeline()
        pipe.multi()
        pipe.mget(assigned_metrics)
        for i, metric_name in enumerate(assigned_metrics):
            pipe.delete(metric_name)
        raw_assigned = pipe.execute()[0]

        # Distill timeseries strings into lists
        # store abnormal data point to mysql database
        cursor = self.mysql_conn.cursor()
        sql = 'INSERT into t_abnormal(time,data) VALUES (%s,%s);'
        for i, metric_name in enumerate(assigned_metrics):
            self.check_if_parent_is_alive()

            raw_series = raw_assigned[i]
            unpacker = Unpacker(use_list=False)
            unpacker.feed(raw_series)
            timeseries = list(unpacker)

            for j, datapoint in enumerate(timeseries):
                cursor.execute(sql, [str(datapoint[0]), str(datapoint)])
                self.mysql_conn.commit()
        cursor.close()

    def run(self):
        """
        Called when process initializes.
        """
        logger.info('started persister')

        while(1):

            now = time()

            # Make sure Redis is up
            try:
                self.redis_conn.ping()
            except:
                logger.error('skyline can\'t connect to redis at socket path %s' % settings.REDIS_SOCKET_PATH)
                sleep(10)
                self.redis_conn = StrictRedis(unix_socket_path=settings.REDIS_SOCKET_PATH)
                continue

            # Discover anomaly metrics
            anomaly_unique_metrics = list(self.redis_conn.smembers(settings.ANOMALY_NAMESPACE + 'unique_metrics'))
            if len(anomaly_unique_metrics) == 0:
                logger.info('No anomaly metrics in redis. Everything is fine.')
                sleep(60)
                continue

            # Spawn processes
            pids = []
            for i in range(1, settings.PERSIST_PROCESSES + 1):
                if i > len(anomaly_unique_metrics):
                    logger.warning('WARNING: skyline is set for more cores than needed.')
                    break

                p = Process(target=self.do_persist, args=(i, anomaly_unique_metrics))
                pids.append(p)
                p.start()

            # Send wait signal to zombie processes
            for p in pids:
                p.join()

            # Sleep if it went too fast
            if time() - now < 5:
                logger.info('persister sleeping due to low run time...')
                sleep(20)
