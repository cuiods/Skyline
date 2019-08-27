## Skyline

Skyline is a real-time anomaly detection system, built to enable passive
monitoring of hundreds of thousands of metrics, without the need to configure a
model/thresholds for each one, as you might do with Nagios. It is designed to be
used wherever there are a large quantity of high-resolution timeseries which
need constant monitoring. Once a metrics stream is set up (from StatsD or
Graphite or other source), additional metrics are automatically added to Skyline
for analysis. Skyline's easily extendible algorithms automatically detect what
it means for each metric to be anomalous. After Skyline detects an anomalous
metric, it surfaces the entire timeseries to the webapp, where the anomaly can be
viewed and acted upon.

## Install

1. `sudo pip install -r requirements.txt` for the easy bits

2. Install numpy, scipy, pandas, patsy, statsmodels, msgpack_python in that
order.

2. You may have trouble with SciPy. If you're on a Mac, try:

* `sudo port install gcc48`
* `sudo ln -s /opt/local/bin/gfortran-mp-4.8 /opt/local/bin/gfortran`
* `sudo pip install scipy`

3. `cp src/settings.py.example src/settings.py`

4.  Edit settings.py according to explanations.

5.  Add directories: 

``` 
sudo mkdir /var/log/skyline
sudo mkdir /var/run/skyline
sudo mkdir /var/log/redis
sudo mkdir /var/dump/
```

6. Download and install the latest Redis release

7. Download and install the latest MySQL release, import tables in `skyline/utils/skyline.sql`, 
configure `MYSQL_` in `settings.py`

8. Start 'er up

* `cd skyline/bin`
* `sudo redis-server redis.conf`
* `sudo ./horizon.d start`
* `sudo ./analyzer.d start`

9. Check the log files to ensure things are running.

### Gotchas

* If you already have a Redis instance running, it's recommended to kill it and
restart using the configuration settings provided in bin/redis.conf

* Be sure to create the log directories.

### Hey! Nothing's happening!
Of course not. You've got no data! For a quick and easy test of what you've 
got, run this:
```
cd utils
python seed_data.py
```
This will ensure that the Horizon
service is properly set up and can receive data. For real data, you have some 
options - see [wiki](https://github.com/etsy/skyline/wiki/Getting-Data-Into-Skyline)

Once you get real data flowing through your system, the Analyzer will be able
start analyzing for anomalies!

### Alerts
Skyline can alert you! In your settings.py, add any alerts you want to the ALERTS
list, according to the schema `(metric keyword, strategy, expiration seconds)` where
`strategy` is one of `smtp`, `hipchat`, or `pagerduty`. You can also add your own
alerting strategies. For every anomalous metric, Skyline will search for the given
keyword and trigger the corresponding alert(s). To prevent alert fatigue, Skyline
will only alert once every <expiration seconds> for any given metric/strategy
combination. To enable Hipchat integration, uncomment the python-simple-hipchat
line in the requirements.txt file.

Besides, abnormal points will be stored in mysql database.

### How to change anomaly detection methods.
You can use different algorithms in skyline. `ALGORITHMS` in `settings.py` are the algorithms that the Analyzer will run. To add a new algorithm,
you must both define the algorithm in algorithms.py and add its name there. `CONSENSUS` is the number of algorithms that must return True before a metric is
classified as anomalous.
