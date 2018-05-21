# numpy, ciso8601, plotly, matplotlib, tqdm, scipy, sklearn, scikits.bootstrap, pandas, joblib, statsmodels, h5py
import multiprocessing
import numpy as np
import ciso8601
import os
import plotly.exceptions as pyex
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime as dt
from scipy.stats.mstats import gmean
from sklearn.preprocessing import Imputer
import scipy.stats
import scipy as sp
from enum import Enum
from collections import Counter
import time
import scikits.bootstrap as bootstrap
from utils.CDR_csv_reader import *
from utils.feature_extration import *
from utils.plot_utils import plot_utils
from joblib import Parallel, delayed
import statsmodels.stats.multitest
import random
import h5py
import traceback

# The type of plot to graph, Ranked or Classic
class PlotType(Enum):
    Classic = 1,
    RankedLine = 2,

class TimeSpanSelected(Enum):
    # Single date such as 20. Oct 2009
    SingleDate = 1
    # Single weekday such as Monday, Tuesday .. etc
    Weekday = 2
    # Show the days relative to the onset: -3 -2 -1  0  1  2  3
    AllDays = 3
    # Shows a long range of actual days, regardless of onsetdate
    LongTerm = 4

NUMBER_OF_CPU_CORES = 1 #multiprocessing.cpu_count()
PARALLEL = Parallel(n_jobs=NUMBER_OF_CPU_CORES)
# Used for selecting healthy individuals for comparison in ranked line.
# Users may now be sick with this period to be selected for comparison.
EXCLUSION_PERIOD_FOR_HEALTHY_USERS = range(-14, 15)

# Filter out uids if they have too few records for the onset week.
NUMBER_OF_ACTIVE_DAYS_IN_ONSET_WEEK = 3

# Only regenerate the graph if it is more than a week old
#REFRESH_TIME_FOR_GRAPHS = 86400 * 7
REFRESH_TIME_FOR_GRAPHS = 0
# Macro used to read huge csv files in chunks

PLOT_TYPE = PlotType.RankedLine
TIME_SPANS = [TimeSpanSelected.AllDays]

# File selection macros
#USE_CONTROL_GROUP = False
USE_CONTROL_GROUP = True
CONTROL_GROUP_TO_USE = 0 # Pick 0-2 for the control group to use
ONSET_AND_CONTROL_GROUP_MAPPING_FILE = 'healthy_sick_pair_uid_mapping.csv'

EXTRA_LOCATION_DATA_FILE = '../../inputData/large/g1sick_gprs_loc.csv' # GPRS data locs for g1 sick
ONSET_DATA_FILE = '../../inputData/large/FULLonset.csv'

G1_CONTROL_FILE = '../../inputData/large/g1calls_control_%d.csv' % CONTROL_GROUP_TO_USE
G1_DIAGNOSED_FILE = '../../inputData/large/g1calls_diagnosed.csv'
LARGE_DATA_FILE = '../../inputData/large/FULLg1sickcalls.csv'  # Only sick individuals from G1
SMALL_DATA_FILE = '../../inputData/small/SMALLg1sickcalls.csv' # Subset of sick individuals from G1

CALL_DATA_FILE, FILE_IN_USE = (LARGE_DATA_FILE, "large")
#CALL_DATA_FILE, FILE_IN_USE = (SMALL_DATA_FILE, "small")

# Macros for exporting data to h5
EXPORT_TO_H5 = False
#EXPORT_TO_H5 = True
H5_DATA_PATH = 'h5Files/' + FILE_IN_USE +'/%d_hour_bins/%s/'
H5_DATA_FILE = '%d_features_%d_days.h5'
SLIDING_WINDOW_SIZE = 7
TRUE_LABEL_RANGE = range(-1, 3)

# List of bin sizes to run
HOURS_PER_BIN_LIST = [4,8,24,48,72,96]

# Used to for confidence intervals and significance test.
SIGNIFICANCE_ALPHA = 0.05

FILE_EXTENSIONS = ["pdf"]

# One Week
#BEFORE_ONSET_PERIOD = range(-73,-3)
#ONSET_PERIOD = range(-3,4)
#AFTER_ONSET_PERIOD = range(4,74)

# Two weeks
#BEFORE_ONSET_PERIOD = range(-77, -7)
#ONSET_PERIOD = range(-7, 8)
#AFTER_ONSET_PERIOD = range(8, 78)

# Four weeks
BEFORE_ONSET_PERIOD = range(-84,-14)
ONSET_PERIOD = range(-14,15)
AFTER_ONSET_PERIOD = range(15,85)


MIN_ONSET_DATE = dt.timedelta(min(ONSET_PERIOD))
MAX_ONSET_DATE = dt.timedelta(max(ONSET_PERIOD))

LONG_TERM_START_DATE = dt.date(2009, 4, 10)
LONG_TERM_END_DATE = dt.date(2010, 5, 10)

LONG_TERM_DATE_RANGE = [
    LONG_TERM_START_DATE + dt.timedelta(i) 
    for i in range((LONG_TERM_END_DATE - LONG_TERM_START_DATE).days + 1)
] 

# Binned user dict is the top most dict
# top most key is the bin size used 
binned_user_day_feature_dicts = None
# User day feature dict is the next layer
# used by plotters and day_exists
user_day_feature_dict = None
# Dates of onset and their uids 
day_uids_onset = None

onset_date_base_comp_ids = None

# Used to collect statistics
# All of the values for each day per user.
SHOULD_COLLECT_STATS = False
sick_uid_day_values_stats = []
healthy_uid_day_values_stats = []
uid_day_values_stats = None

# =====================================
#               Utils
# =====================================
def no_progress_bar(collection, desc=""):
    return collection
def datetime_to_datestr(timestamp):
    return timestamp.strftime("%d.%b.%y")
def datetime_to_weekdaystr(timestamp):
    return timestamp.strftime("%A")

def bootstrap_mean_confidence_interval(data):
    if data == []:
        return np.nan, np.nan
    low, high = bootstrap.ci(data=data, statfunction=np.nanmean, method='bca', n_samples=10000, alpha=SIGNIFICANCE_ALPHA)
    mean = np.nanmean(data)
    return mean, high - mean, mean - low

def collect_adjacent_days(days, uid, bin_id, is_nan_feature, calc_callback):
    counters = []
    for adjacent_day in days:
        if is_nan_feature or day_exists(uid, adjacent_day, bin_id):
            callback = calc_callback[0]
            params = calc_callback[1:]
            measure = callback(uid, adjacent_day, bin_id, user_day_feature_dict, *params)
            counters.append(measure)
    return counters
def collect_data_parallel(uid, bin_id, start_date, end_date, past_day_count, future_day_count, is_nan_feature, calc_callback):
    uid_day_counters = []
    single_day = dt.timedelta(1)
    today = start_date
    while today <= end_date:
        if is_nan_feature or day_exists(uid, today, bin_id):
            today_value = collect_adjacent_days([today], uid, bin_id, is_nan_feature, calc_callback)
    
            past_days = ((today - dt.timedelta(7 * i)) for i in xrange(1, past_day_count + 1))
            past_values = collect_adjacent_days(past_days, uid, bin_id, is_nan_feature, calc_callback)

            future_days = ((today + dt.timedelta(7 * i)) for i in xrange(1, future_day_count + 1))
            future_values = collect_adjacent_days(future_days, uid, bin_id, is_nan_feature, calc_callback)
            counters = (past_values, today_value, future_values)
        else:
            counters = ([], [np.nan], [])
        uid_day_counters.append(counters)
        today = today + single_day
    return uid_day_counters
def collect_data(uids, bin_id, start_date, end_date, past_day_count, future_day_count, is_nan_feature, calc_callback, progress_bar=tqdm, do_parallel=True):
    if do_parallel:
        uid_day_values = PARALLEL(
            delayed(collect_data_parallel)
            (uid, bin_id, start_date, end_date, past_day_count, future_day_count, is_nan_feature, calc_callback)
            for uid in progress_bar(uids, desc="Collecting Values")
        )
    else:
        uid_day_values = []
        for uid in progress_bar(uids, desc="Collecting Values"):
            values = collect_data_parallel(uid, bin_id, start_date, end_date, past_day_count, future_day_count, is_nan_feature, calc_callback)
            uid_day_values.append(values)
    if SHOULD_COLLECT_STATS:
        uid_stats = []
        for uid_days in uid_day_values:
            uid_stats.append([today for _, [today], _ in uid_days])
        uid_day_values_stats.extend(uid_stats)
    return uid_day_values

def significance_fwer_correction(p_values):
    N = len(p_values)
    p_values = np.sort(p_values)
    i = np.arange(1, N + 1)
    q = SIGNIFICANCE_ALPHA
    below = p_values < (q * i / N)
    max_below = np.max(np.where(below[0]))
    plt.plot(i, p_values, 'b.', label='$p(i)$')
    plt.plot(i, q * i / N, 'r', label='$q i / N$')
    plt.xlabel('$i$')
    plt.ylabel('$p$')
    plt.legend()
    plt.savefig('fdr_corr')
def significance_fdr_correction(plot_info, onset_date, p_values,d_values, sick_line, healthy_line, test_type):
    corrected_p_vals = statsmodels.stats.multitest.multipletests(
        p_values, alpha=SIGNIFICANCE_ALPHA, method='fdr_bh')

    path = get_file_name(plot_info, "significance", onset_date)
    # Log to file
    with open(path + ("_%s.tex" % test_type), 'w') as tex_file:
        tex_file.write("%% Significance: %s (%s)\n" % (plot_info["title"], path))
        tex_file.write("\\begin{tabular}{c|c|c|c|c|c|c}\n")
        tex_file.write("%3s & %5s & %7s & %10s & %10s & %10s & %6s\\\\\n" %
                       ("Day", "Sick", "Healthy", "%s D-vals" %test_type, "%s p-vals" % test_type, "FDR p-vals", "Reject"))
        tex_file.write("\\hline\n")
        for i in range(len(ONSET_PERIOD)):
            tex_file.write("%3d & %4.3f & %7.3f & %7.3f %10.2e & %10.2e & %6s\\\\\n" % (
                ONSET_PERIOD[0] + i,  # days from diagnosis
                sick_line[0][i],  # sick value
                healthy_line[0][i],  # healthy value
                d_values[i],  # test statistic
                p_values[i],  # original p-value
                corrected_p_vals[1][i],  # corrected p-value
                corrected_p_vals[0][i])  # reject array value
            )
        tex_file.write("\\end{tabular}")
def significance_test_wrapper(test_func, sick_counters, healthy_counters):
    sick_days = np.array(sick_counters).T
    healthy_days = np.array(healthy_counters).T
    def remove_missing_data(counters):
        new_ranks = []
        for uid_ranks in counters:
            new_ranks.append([v for v in uid_ranks if not np.isnan(v)])
        return np.array(new_ranks).T

    sick_days = remove_missing_data(sick_days)
    healthy_days = remove_missing_data(healthy_days)
    p_values, d_values = [], []

    for i in range(len(ONSET_PERIOD)):
        test_result = test_func(sick_days[i], healthy_days[i])
        p_values.append(test_result.pvalue)
        d_values.append(test_result.statistic)

    return p_values, d_values

def significance_ks_test(sick_counters, healthy_counters):
    return significance_test_wrapper(scipy.stats.ks_2samp, sick_counters, healthy_counters)
def significance_mann_whitney_u_test(sick_counters, healthy_counters):
    return significance_test_wrapper(scipy.stats.mannwhitneyu, sick_counters, healthy_counters)

def should_regenerate_file(file_names):
    for file_name in file_names:
        if not (os.path.exists(file_name) and
                time.time() - REFRESH_TIME_FOR_GRAPHS < os.path.getmtime(file_name)):
            return True
    return False
def get_graph_title(plot_info, num_onset_dates, number_of_users, onset_date):
    hours_per_bin = plot_info["hours_per_bin"]
    bin_id = plot_info["bin_id"]
    time_selected = plot_info["time_selected"]

    title = plot_info["title"] + "<br>"
    if time_selected == TimeSpanSelected.LongTerm:
        title += "Data from %d individuals" % number_of_users
    elif time_selected == TimeSpanSelected.AllDays:
        title += "Combined data from %d dates of diagnosis" % num_onset_dates
    else:
        if time_selected == TimeSpanSelected.Weekday:
            title += "All %d individuals diagnosed on %ss (%d dates)" % (
                number_of_users, onset_date.strftime("%A"), num_onset_dates)
        elif time_selected == TimeSpanSelected.SingleDate:
            title += "All %d individuals diagnosed on %s" % (
                number_of_users, onset_date.strftime("%A, %d. %B %Y"))
    if hours_per_bin < 24:
        title += "<br>Data from time %d:00 to %d:00 " % (
            hours_per_bin * bin_id, hours_per_bin * (bin_id + 1))

    return title
def get_file_name(plot_info, ending, onset_date=None):
    bin_id = plot_info["bin_id"]
    hours_per_bin = plot_info["hours_per_bin"]
    time_selected = plot_info["time_selected"]

    plot_type_str = "classic"
    if PLOT_TYPE == PlotType.RankedLine:
        plot_type_str = "ranked_line"
    path = "graphs_%s/%f_alpha_%s/" % (
        FILE_IN_USE,
        SIGNIFICANCE_ALPHA,
        ending
    )
    if time_selected == TimeSpanSelected.LongTerm:
        path += "%dh_%s_long_term/%s_to_%s" % (
            hours_per_bin,
            plot_type_str,
            LONG_TERM_START_DATE.strftime("%d.%b.%y"),
            LONG_TERM_END_DATE.strftime("%d.%b.%y")
        )
    else:    
        path += "%dh_%s_%d_days/%s" % (
            hours_per_bin,
            plot_type_str,
            len(ONSET_PERIOD),
            ("single_date" if time_selected == TimeSpanSelected.SingleDate else
             "weekday" if time_selected == TimeSpanSelected.Weekday else
             "relative_days")
        )

    day_str = ""
    if not time_selected == TimeSpanSelected.AllDays and \
       not time_selected == TimeSpanSelected.LongTerm:
        plot_folder = plot_info["folder_name"] if "folder_name" in plot_info else None
        if plot_folder:
            path += "/" + plot_folder
        path += "/" + plot_info["plot_name"]
        day_str = "_%s" % (onset_date.strftime(
            "%d-%b-%y" if time_selected == TimeSpanSelected.SingleDate else "%A").lower())

    if not os.path.exists(path):
        os.makedirs(path)
    return '%s/%s%s_bin%d' % (path, plot_info["plot_name"], day_str, bin_id)
def plot_names(plot_info, onset_date=None):
    file_names = []
    for ending in FILE_EXTENSIONS:
        path = get_file_name(plot_info, ending, onset_date)
        file_names.append("%s.%s" % (path, ending))
    return file_names
def day_exists(uid, day, bin_id):
    if uid not in user_day_feature_dict:
        print('USER NOT FOUND, We have a problem', uid)
        return False
    elif day not in user_day_feature_dict[uid]:
        #print("day not found for uid")
        return False
    elif bin_id not in user_day_feature_dict[uid][day]:
        return False
    else:
        return True
def save_figure(fig, plot_info):
    if FILE_IN_USE == 'large' or USE_CONTROL_GROUP:
    	plot_url = py.plot(fig, filename=plot_info["file_names"][0])

    for file_name in plot_info["file_names"]:
        print("Saving:", file_name)
        py.image.save_as(fig, filename=file_name)

# =====================================
#            Plot Classic
# =====================================
def plot_classic_layout(before_bar, onset_line, after_bar, plot_info, number_of_users, number_of_dates, period, past_day_count, future_day_count, onset_date):
    hours_per_bin = plot_info["hours_per_bin"]
    bin_id = plot_info["bin_id"]
    time_selected = plot_info["time_selected"]

    shapes = [{
        "line": {
            "color": "rgb(100, 100, 100)",
            "dash": "dot",
            "width": 2
        },
        "type": "line",
        "x0": 0.5,
        "x1": 0.5,
        "xref": "paper",
        "y0": 0,
        "y1": 1,
        "yref": "paper"
    }]
    traces = [
        {
            "name": "Previous %d same days of the week" % past_day_count,
            "x": period,
            "y": before_bar[0],
            "type": "bar",
            "marker": {"color": "rgb(20, 106, 166)"},  # Blue
            "error_y": {
                "array": before_bar[1], # Confidence low val 
                "arrayminus": before_bar[2], # Confidence high val 
                "symmetric": False,
                "type": "data",
                "visible": True,
                "thickness": 3,
                "width": 6
            },
        },
        {
            "name": "Current day value",
            "x": period,
            "y": onset_line[0],
            "type": "scatter",
            "line": {
                "color": "rgb(255, 127, 34)",  # Orange
                "shape": "spline",
                "width": 3
            },
            "marker": {
                "color": "rgb(255, 127, 14)",
                #"line": {"color": "rgb(255, 127, 14)"},
                "size": 8
            },
            "error_y": {
                "array": onset_line[1], # Confidence low val 
                "arrayminus": onset_line[2], # Confidence high val 
                "symmetric": False,
                "color": "rgb(255, 127, 14)",
                "type": "data",
                "thickness": 3,
                "width": 6
            },
        },
        {
            "name": "Next %d same days of the week" % future_day_count,
            "x": period,
            "y": after_bar[0],
            "type": "bar",
            "marker": {"color": "rgb(51, 179, 51)"},  # Green
            "error_y": {
                "array": after_bar[1], # Confidence low val 
                "arrayminus": after_bar[2], # Confidence high val 
                "symmetric": False,
                "type": "data",
                "visible": True,
                "thickness": 3,
                "width": 6
            },
        }
    ]

    xaxis= {
        "autorange": True,
        "nticks": 40,
        "tickformat": "%A" if len(period) < 8 else "%a",
        "tickmode": "linear",
        "type": "date"
    }
    if time_selected == TimeSpanSelected.LongTerm:
        xaxis["title"] = "Days between %s and %s" % (
            onset_date[0].strftime("%d. %b %Y"),
            onset_date[1].strftime("%d. %b %Y")
        )
        xaxis["tickmode"] = "auto"
        xaxis["tickformat"] = ""
    elif time_selected == TimeSpanSelected.SingleDate:
        xaxis["title"] = "Days around diagnosis (%s)" % onset_date.strftime("%A, %d. %B %Y")
    else:
        xaxis["title"] = "Days around diagnosis (%s)" % onset_date.strftime("%A")

    data = go.Data(traces)
    layout = {
        "bargap": 0.2,
        "bargroupgap": 0,
        "barmode": "group",
        "legend": {
            "orientation": "h",
            "x": 0,
            "y": -0.25,
        },
        "font": {
            "size": 14
        }, 
        "margin": {"r": 20, "t": 100, "b": 40, "l": 70},
        "title": get_graph_title(plot_info, number_of_dates, number_of_users, onset_date),
        "xaxis": xaxis,
        "yaxis": {
            "autorange": True,
            "title": plot_info["yaxis"],
            "type": "linear"
        },
        "shapes": shapes if len(period) > 7 and time_selected != TimeSpanSelected.LongTerm else []
    }
    fig = go.Figure(data=data, layout=layout)
    return fig

def calc_bootstrap_for_classic(period_len, before_counter, onset_counter, after_counter):
    print("Running bootstrapping")
    counters =  PARALLEL(
        delayed(bootstrap_mean_confidence_interval)(day_counter) 
        for day_counter in tqdm(before_counter + onset_counter + after_counter) 
    )
    print("Bootstrapping done")
    before_counter = counters[:period_len]
    before_counter = np.array(before_counter).T

    onset_counter = counters[period_len : 2 * period_len]
    onset_counter = np.array(onset_counter).T

    after_counter = counters[2 * period_len:]
    after_counter = np.array(after_counter).T

    return before_counter, onset_counter, after_counter

def plot_classic_for_onsets(onset_and_uids, is_nan_feature, plot_info, calc_callback):
    bin_id = plot_info["bin_id"]

    past_day_count = 10
    future_day_count = 10
    onset_period_len = len(ONSET_PERIOD)
    before_counter = [[] for _ in range(onset_period_len)]
    after_counter = [[] for _ in range(onset_period_len)]
    onset_counter = [[] for _ in range(onset_period_len)]

    for onset_date, uids in onset_and_uids:
        start_date = MIN_ONSET_DATE + onset_date 
        end_date = MAX_ONSET_DATE + onset_date
        uid_counters = collect_data(uids, bin_id, start_date, end_date, past_day_count, future_day_count, is_nan_feature, calc_callback)
        for counters in uid_counters:
            for day_index in range(onset_period_len):
                past, cur, future = counters[day_index]
                before_counter[day_index].extend(past)
                after_counter[day_index].extend(future)
                onset_counter[day_index].extend(cur)

    before_counter, onset_counter, after_counter = calc_bootstrap_for_classic(onset_period_len, before_counter, onset_counter, after_counter)
    number_of_users = sum([len(uids) for _, uids in onset_and_uids])
    number_of_dates = len(onset_and_uids)
    fig = plot_classic_layout(
        before_counter, onset_counter, after_counter, 
        plot_info, number_of_users, number_of_dates, ONSET_PERIOD,
        past_day_count, future_day_count, onset_date
    )

    save_figure(fig, plot_info)

def plot_classic_for_long_term(uids, start_date, end_date, plot_info, is_nan_feature, calc_callback):
    bin_id = plot_info["bin_id"]

    past_day_count = 10
    future_day_count = 10
    long_term_len = len(LONG_TERM_DATE_RANGE)
    before_counter = [[] for _ in range(long_term_len)]
    after_counter = [[] for _ in range(long_term_len)]
    onset_counter = [[] for _ in range(long_term_len)]

    uid_counters = collect_data(uids, bin_id, start_date, end_date, past_day_count, future_day_count, is_nan_feature, calc_callback)
    for counters in uid_counters:
        for day_index in range(long_term_len):
            past, cur, future = counters[day_index]
            before_counter[day_index].extend(past)
            after_counter[day_index].extend(future)
            onset_counter[day_index].extend(cur)

    before_counter, onset_counter, after_counter = calc_bootstrap_for_classic(long_term_len, before_counter, onset_counter, after_counter)
    number_of_users = len(uids)
    number_of_dates = 1
    fig = plot_classic_layout(
        before_counter, onset_counter, after_counter, 
        plot_info, number_of_users, number_of_dates, LONG_TERM_DATE_RANGE,
        past_day_count, future_day_count, (start_date, end_date)
    )
    save_figure(fig, plot_info)

# =====================================
#            Plot Ranked Line
# =====================================
def plot_ranked_line_layout(sick_line, healthy_line, plot_info, num_onset_dates, num_sick_users, num_healthy_users, onset_date):
    hours_per_bin = plot_info["hours_per_bin"]
    bin_id = plot_info["bin_id"]
    time_selected = plot_info["time_selected"]

    if time_selected == TimeSpanSelected.LongTerm:
        start_date, end_date = onset_date
        onset_days = LONG_TERM_DATE_RANGE
    else: 
        onset_days = [onset_date + dt.timedelta(x) for x in ONSET_PERIOD]
        if time_selected == TimeSpanSelected.AllDays:
            onset_days = ONSET_PERIOD

    def get_trace(line, num_users, name, symbol_type):
        name_type = "samples"
        if time_selected == TimeSpanSelected.LongTerm or num_onset_dates == 1 or USE_CONTROL_GROUP:
            name_type = "individuals"

        return {
            "name": "%s (%d %s)" % (name, num_users, name_type),
            "x": onset_days,
            "y": line[0],
            "type": "scatter",
            "line": {
                "shape": "spline",
                "width": 3.5
            },
            "marker": {
                "size": 7.5,
                "symbol": symbol_type,
            },
            "mode": "lines+markers",
            "error_y": {
                "array": line[2], # Confidence high val 
                "arrayminus": line[1], # Confidence low val 
                "symmetric": False,
                "type": "data",
                "visible": True,
                "thickness": 3,
                "width": 6
            },
        }

    traces = [
        get_trace(healthy_line, num_healthy_users, "Control", "diamond"),
        get_trace(sick_line, num_sick_users, "Diagnosed", "circle"),
    ]
    data = go.Data(traces)
    xaxis = {
        "autorange": True,
        "tickmode": "linear",
        "dtick": 1 if len(ONSET_PERIOD) <= 15 else 2,
    }
    shapes = []
    if time_selected == TimeSpanSelected.AllDays:
        xaxis["title"] = "Days from diagnosis (0 is DoD)"
    elif time_selected == TimeSpanSelected.LongTerm:
        xaxis["autorange"] = True
        xaxis["dtick"] = 2
        xaxis["nticks"] = 0
        xaxis["tickmode"] = "auto"
        xaxis["title"] = "Days between %s and %s" % (
            onset_date[0].strftime("%d. %b %Y"),
            onset_date[1].strftime("%d. %b %Y")
        )
    else:
        xaxis["nticks"] = 40
        xaxis["tickformat"] = "%A" if len(ONSET_PERIOD) < 8 else "%a"
        xaxis["tickmode"] = "linear"
        xaxis["type"] = "date"
        if time_selected == TimeSpanSelected.SingleDate:
            date_str = onset_date.strftime("%A, %d. %B %Y")
        else:
            date_str = onset_date.strftime("%A")
        xaxis["title"] = "Days around diagnosis (%s)" % date_str
        shapes = [{
            "line": {
                "color": "rgb(100, 100, 100)",
                "dash": "dot",
                "width": 2
            },
            "type": "line",
            "x0": 0.5,
            "x1": 0.5,
            "xref": "paper",
            "y0": 0,
            "y1": 1,
            "yref": "paper"
        }]

    y_range = [0.3, 0.7]
    y_dtick = 0.05
    if min(sick_line[0]) < 0.3 or max(sick_line[0]) > 0.7:
        y_range = [0, 1]
        y_dtick = 0.1

    layout = {
        "legend": {
            "orientation": "h",
            "x": 0,
            "y": -0.25,
        },
        "font": {
            "size": 14
        }, 
        "margin": {"r": 20, "t": 100, "b": 40, "l": 70},
        "title": get_graph_title(plot_info, num_onset_dates, num_sick_users + num_healthy_users, onset_date),
        "xaxis": xaxis,
        "yaxis": {
            "autorange": False,
            "range": y_range,
            "dtick": y_dtick,
            "title": "Normalized Fractional Rank",
            "type": "linear"
        },
        "shapes": shapes,
    }
    fig = go.Figure(data=data, layout=layout)
    return fig

def get_individual_ranks(uids, start_date, end_date, bin_id, past_day_count, future_day_count, is_nan_feature, calc_callback, progress_bar=tqdm, do_parallel=True):
    onset_counters_per_uid = collect_data(uids, bin_id, start_date, end_date, past_day_count, future_day_count, is_nan_feature, calc_callback, progress_bar, do_parallel)

    individual_uid_ranks = []
    for uid_index in progress_bar(xrange(len(uids)), desc="Calculating Ranks"):
        day_ranks = []
        for past_values, today_value, _ in onset_counters_per_uid[uid_index]:
            if past_values == [] or today_value == [np.nan]:
                day_ranks.append(np.nan)
                continue
            values = today_value + past_values
            ranked = sp.stats.rankdata(values)
            normalized_rank = (ranked[0] - 1) / (len(values) - 1)
            day_ranks.append(normalized_rank)
        individual_uid_ranks.append(day_ranks)
    return individual_uid_ranks
def get_individual_ranks_parallel(total, bin_id, onset_date, sick_uids, healthy_uids, is_nan_feature, calc_callback):
    start_date = MIN_ONSET_DATE + onset_date 
    end_date = MAX_ONSET_DATE + onset_date
    past_day_count = 10
    future_day_count = 0

    global uid_day_values_stats
    uid_day_values_stats = healthy_uid_day_values_stats
    healthy = get_individual_ranks(sick_uids, start_date, end_date, bin_id, past_day_count, future_day_count, is_nan_feature, calc_callback, no_progress_bar, False)
    uid_day_values_stats = sick_uid_day_values_stats
    sick =  get_individual_ranks(healthy_uids, start_date, end_date, bin_id, past_day_count, future_day_count, is_nan_feature, calc_callback, no_progress_bar, False)
    
    return (healthy, sick)
def get_ranked_data_for_onsets(onset_date_base_comp_ids, plot_info, is_nan_feature, calc_callback):
    bin_id = plot_info["bin_id"]
    total = len(onset_date_base_comp_ids)
    if FILE_IN_USE == 'large' and not SHOULD_COLLECT_STATS:
        results = PARALLEL(
            delayed(get_individual_ranks_parallel)
            (total, bin_id, onset_date, sick_uids, healthy_uids, is_nan_feature, calc_callback)
            for onset_date, sick_uids, healthy_uids, in tqdm(onset_date_base_comp_ids))
    else:
        results = [
            get_individual_ranks_parallel(total, bin_id, onset_date, sick_uids, healthy_uids, is_nan_feature, calc_callback)
            for onset_date, sick_uids, healthy_uids, in tqdm(onset_date_base_comp_ids)
        ]
    [sick_counters, healthy_counters] = map(list, zip(*results))
    sick_counters = [arr for arr3d in sick_counters for arr in arr3d]
    healthy_counters = [arr for arr3d in healthy_counters for arr in arr3d]
    return sick_counters, healthy_counters

def create_cdf(rank_list, transform_func=None, resolution=0.01):
    # Nr. of ranks
    total = float(len(rank_list))
    # Return an array of tuples on the form (rank, count)
    data = Counter(rank_list).items()
    # Sort the array based on the ranks
    data = sorted(data, key=lambda x: x[0])
    # Sum the counts
    y_sum = float(sum([val for _, val in data]))

    # Construct list of y's for CDF by adding up the values and dividing by the total nr.
    x = np.arange(resolution, 1+resolution, resolution)
    y = []
    lower = -resolution
    cum = 0
    for upper in x:
        # Sum up all the counters within this resolution window
        cum += sum([val for key, val in data if lower < key <= upper])
        s = cum / y_sum
        if transform_func is not None:
            s = transform_func(s)
        y.append(s)
        lower = upper
    
    return x, y
def find_outline(x_s, y_s, extreme_func):
    return x_s[0], [extreme_func(y_arr) for y_arr in np.array(y_s).T]

def create_distribution_figure_multiple_days(plot_info, onset_date, sick_counters, healthy_counters, ks_d_values):
    path = get_file_name(plot_info, "distribution", onset_date)
    if not os.path.exists(path):
        os.makedirs(path)

    sick_x_s, sick_y_s, healthy_x_s, healthy_y_s, conf_x_s, conf_y_s = [], [], [], [], [], []
    
    title = plot_info["plot_name"]
    day_list = []
    if title == "unique_locations_visited":
        day_list = range(12,20)
    elif title == "calls_count_outgoing":
        day_list = [13, 15, 16, 17, 18, 21]
    elif title == "mean_call_duration_both":
        day_list = range(13,18)
    else:
        day_list = range(10,19)
    for day_index in day_list:
        sick_ranks_today = sick_counters[day_index]
        healthy_ranks_today = healthy_counters[day_index]
        d_val = ks_d_values[day_index]
        x, y = create_cdf(sick_ranks_today)
        sick_x_s.append(x)
        sick_y_s.append(y)
        
        x, y = create_cdf(healthy_ranks_today)
        healthy_x_s.append(x)
        healthy_y_s.append(y)

        u_x, u_y = create_cdf(healthy_ranks_today, lambda val: min(val + d_val, 1))
        l_x, l_y = create_cdf(healthy_ranks_today, lambda val: max(val - d_val, 0))
        conf_x_s.append(u_x)
        conf_y_s.append(u_y)
        conf_x_s.append(l_x)
        conf_y_s.append(l_y)

    max_sick_x, max_sick_y = find_outline(sick_x_s, sick_y_s, max)
    min_sick_x, min_sick_y = find_outline(sick_x_s, sick_y_s, min)
    max_healthy_x, max_healthy_y = find_outline(healthy_x_s, healthy_y_s, max)
    min_healthy_x, min_healthy_y = find_outline(healthy_x_s, healthy_y_s, min)
    max_conf_x, max_conf_y = find_outline(conf_x_s, conf_y_s, max)
    min_conf_x, min_conf_y = find_outline(conf_x_s, conf_y_s, min)

    full_path = '%s/multiple.pdf' % (path)
    # Sick lines
    plt.plot(max_sick_x, max_sick_y, 'r-', label='Diagnosed')
    plt.plot(min_sick_x, min_sick_y, 'r-')
    # Healthy lines
    plt.plot(max_healthy_x, max_healthy_y, 'b--', label='Control')
    plt.plot(min_healthy_x, min_healthy_y, 'b--')
    # Confidence lines
    plt.plot(max_conf_x, max_conf_y, 'k-.', label='Confidence Band')
    plt.plot(min_conf_x, min_conf_y, 'k-.')
    # Fill between lines
    plt.fill_between(max_sick_x, max_sick_y, min_sick_y, color='red', alpha='0.9')
    plt.fill_between(max_healthy_x, max_healthy_y, min_healthy_y, color='blue', alpha='0.9')
    plt.fill_between(max_conf_x, max_conf_y, min_conf_y, color='black', alpha='0.6')
    #i = 2
    #for _ in day_list[1:]:
    #    plt.plot(conf_x_s[0+i], conf_y_s[0+i], 'k-.')
    #    plt.plot(conf_x_s[1+i], conf_y_s[1+i], 'k-.')
    #    i += 2
    plt.xlabel("Normalized fractional ranking")
    plt.ylabel("Percentage of population")
    plt.legend()
    plt.xticks(np.arange(0, 1.05, 0.1))
    plt.yticks(np.arange(0, 1.05, 0.1))
    plt.title(plot_info["title"] + "\nCumulative distribution on days %s" % str([day - 14 for day in day_list])[1:-1])
    plt.savefig(full_path)
    plt.clf()
def create_distribution_plots_shared_y_axis(plot_info, onset_date, sick_counters, healthy_counters, ks_d_values):
    path = get_file_name(plot_info, "distribution", onset_date)
    if not os.path.exists(path):
        os.makedirs(path)

    sick_x_s, sick_y_s, healthy_x_s, healthy_y_s, conf_x_s, conf_y_s = [], [], [], [], [], []

    title = plot_info["plot_name"]
    day_list = []
    if title == "unique_locations_visited":
        day_list = range(12,20) # 8 days (3x3)
    elif title == "calls_count_outgoing":
        day_list = [13, 15, 16, 17, 18, 21] # 6 days (3x2)
    elif title == "mean_call_duration_both":
        day_list = range(13,18) # 5 days (3x2)
    else:
        day_list = range(11,19) # 8 days (3x3)
    
    # Set CDF resolution for smooth lines
    cdf_resolution = 0.01
    # Coefficient for KS Confidence intervals
    coeff = np.sqrt(-0.5 * np.log(SIGNIFICANCE_ALPHA / 2))
    # Iterate through the (hardcoded) significant days for selected feature
    for day_index in tqdm(day_list, desc="Calculating CDFs"):
        sick_ranks_today = sick_counters[day_index]
        healthy_ranks_today = healthy_counters[day_index]

        # Calculate critical value D_alpha for current day
        n = len(sick_ranks_today)
        m = len(healthy_ranks_today)
        d_val = coeff *  np.sqrt(float(n+m) / float(n*m))

        x, y = create_cdf(sick_ranks_today, resolution=cdf_resolution)
        sick_x_s.append(x)
        sick_y_s.append(y)
        
        x, y = create_cdf(healthy_ranks_today, resolution=cdf_resolution)
        healthy_x_s.append(x)
        healthy_y_s.append(y)

        u_x, u_y = create_cdf(healthy_ranks_today, lambda val: min(val + d_val, 1), cdf_resolution)
        l_x, l_y = create_cdf(healthy_ranks_today, lambda val: max(val - d_val, 0), cdf_resolution)
        conf_x_s.append(u_x)
        conf_y_s.append(u_y)
        conf_x_s.append(l_x)
        conf_y_s.append(l_y)

    distance_x_s, distance_y_s, distance_vals = [], [], []
    # Iterate through days to find the max distance between sick and healthy
    for i in range(len(day_list)):
        healthy_y_s_today = healthy_y_s[i]
        sick_y_s_today = sick_y_s[i]
        max_distance = -1
        cdf_index = -1
        # Iterate through x axis (CDF resolution) on current day to find max distance between sick and healthy
        for j in range(int(1 / cdf_resolution)):
            healthy_y_val = healthy_y_s_today[j]
            sick_y_val = sick_y_s_today[j]
            distance = abs(healthy_y_val - sick_y_val)
            if distance > max_distance:
                max_distance = distance
                cdf_index = j
        x_val_healthy = healthy_x_s[i][cdf_index]
        x_val_sick = sick_x_s[i][cdf_index]
        assert x_val_healthy == x_val_sick
        y_val_healthy = healthy_y_s[i][cdf_index]
        y_val_sick = sick_y_s[i][cdf_index]
        distance_x_s.append([x_val_sick, x_val_healthy])
        distance_y_s.append([y_val_sick, y_val_healthy])
        distance_vals.append('%.3f' % max_distance)

    full_path = '%s/shared_axis.pdf' % (path)
    subplot_count = len(day_list)
    subplot_cols = 4 if title == "unique_locations_visited" else (5 if title == "mean_call_duration_both" else 6)
    subplot_rows = int(np.ceil(float(subplot_count) / float(subplot_cols)))

    show_legend = subplot_cols == 4
    fig = plt.figure(figsize=(
        subplot_cols * 3,
        subplot_rows * 4
    ))
    # Construct the main plot
    main_ax = fig.add_subplot(111)
    # Remove the gridlines (frame) on the main plot
    main_ax.spines['left'].set_color('none')
    main_ax.spines['bottom'].set_color('none')
    main_ax.spines['right'].set_color('none')
    main_ax.spines['top'].set_color('none')
    # Remove the ticklabels on the main plot
    main_ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    # Construct the subplots
    axes = [[None for c in range(subplot_cols) if ((c+1) * (r+1)) <= subplot_count] for r in range(subplot_rows)]
    i = 1
    for r in range(subplot_rows):
        if i > subplot_count:
            break
        for c in range(subplot_cols):
            if i > subplot_count:
                break
            ax = fig.add_subplot(subplot_rows, subplot_cols, i)
            ax.tick_params(axis='y', direction='in')
            ax.tick_params(axis='x', direction='in')
            axes[r][c] = ax
            i += 1

    def find_distance_label_pos(x_arr, y_arr):
        x = x_arr[0]
        y_max, y_min = max(y_arr), min(y_arr)
        x_pos, y_pos = -1, -1
        is_label_above = None
        if y_max > 0.75:
            y_pos = y_min - 0.1
            is_label_above = False
        else:
            y_pos = y_max + 0.1
            is_label_above = True
        if x < 0.15:
            x_pos = x + 0.15
        elif x > 0.85:
            x_pos = x - 0.15
        else:
            if is_label_above:
                x_pos = x - 0.15
            else:
                x_pos = x + 0.15
        
        return x_pos, y_pos
    
    # Populate the subplots with data
    days = [day - 14 for day in day_list]
    i = 0
    conf_index = 0
    for r in range(subplot_rows):
        if (i+1) > subplot_count:
            break
        for c in range(subplot_cols):
            if (i+1) > subplot_count:
                break
            ax = axes[r][c]
            ax.set_yticks(np.arange(0, 1.05, 0.2))
            ax.set_xticks(np.arange(0, 1.05, 0.2))
            if c != 0:
                ax.tick_params(axis='y', labelcolor='w')
            if r != (subplot_rows - 1) and not ((r+1) * (c+1) + subplot_cols) > subplot_count:
                ax.tick_params(axis='x', labelcolor='w')
            sick_line = ax.plot(sick_x_s[i], sick_y_s[i], 'r-')
            healthy_line = ax.plot(healthy_x_s[i], healthy_y_s[i], 'b-')
            conf_u_line = ax.plot(conf_x_s[conf_index], conf_y_s[conf_index], 'b-', alpha=0.1)
            ax.plot(conf_x_s[conf_index+1], conf_y_s[conf_index+1], 'b-', alpha=0.1)
            ax.fill_between(conf_x_s[conf_index], conf_y_s[conf_index], conf_y_s[conf_index+1], color=((0, 0, 1, 0.15)))
            dist_line = ax.plot(distance_x_s[i], distance_y_s[i], 'g-')
            dist_label_pos_x, dist_label_pos_y = find_distance_label_pos(distance_x_s[i], distance_y_s[i])
            ax.text(dist_label_pos_x, dist_label_pos_y, distance_vals[i], ha='center', va='center', color='g', weight='bold')
            ax.set_title('Day %d | KS: %.3f' % (days[i], ks_d_values[i]), fontdict={'fontsize': 10}, pad=2)
            i += 1
            conf_index += 2

    #fig, axes = plt.subplots(subplot_rows, subplot_cols, sharex='col', sharey='row', figsize=(8, 11.4))
    #days = [day - 14 for day in day_list]
    #i = 0
    #for r in range(subplot_rows):
    #    for c in range(subplot_cols):
    #        ax = axes[r][c]
    #        if i >= subplot_count:
    #            ax.set_visible(False)
    #            break
    #        ax.plot(sick_x_s[i], sick_y_s[i], 'r-', label='Diagnosed')
    #        ax.plot(healthy_x_s[i], healthy_y_s[i], 'b--', label='Control')
    #        ax.plot(conf_x_s[i], conf_y_s[i], 'k-.', label='Upper confidence')
    #        ax.plot(conf_x_s[i+1], conf_y_s[i+1], 'k-.', label='Lower confidence')
    #        ax.set_title('Distribution on day %d' % days[i])
    #        i += 1
    #plt.suptitle(plot_info["title"] + "\nCumulative distribution on days %s" % str(days).translate(None, "[]"))

    # Set common title and axis labels
    main_ax.set_title(plot_info["title"] + " CDFs", pad=20)
    main_ax.set_xlabel("Normalized fractional ranking")
    main_ax.set_ylabel("Percentage of population")
    if show_legend:
         fig.legend((sick_line[0], healthy_line[0], conf_u_line[0], dist_line[0]), ('Diagnosed', 'Control', 'Confidence', 'Max $\Delta$'), 'lower right')
    #plt.legend()
    #fig.set_size_inches((2, 6))
    plt.savefig(full_path, transparent=True)
    plt.clf()

def create_distribution_figure_parallel(plot_info, onset_date, day_index, sick_counters, healthy_counters, ks_d_values):
    path = get_file_name(plot_info, "distribution", onset_date)
    if not os.path.exists(path):
        os.makedirs(path)

    sick_ranks_today = sick_counters[day_index]
    healthy_ranks_today = healthy_counters[day_index]

    ks_d_val = ks_d_values[day_index]

    x_s, y_s = [], []

    # 2 iterations: 1st for sick ranks, 2nd for healthy ranks
    for rank_list in [sick_ranks_today, healthy_ranks_today]:
        x, y = create_cdf(rank_list)
        x_s.append(x)
        y_s.append(y)

    # Calculate critical value D_alpha
    coeff = np.sqrt(-0.5 * np.log(SIGNIFICANCE_ALPHA / 2))
    n = len(sick_ranks_today)
    m = len(healthy_ranks_today)
    d_val = coeff *  np.sqrt(float(n+m) / float(n*m))

    # Calculate confidence intervals
    data = Counter(healthy_ranks_today).items()
    data = sorted(data, key=lambda x: x[0])
    x = [key for key, _ in data]
    x_s.append(x)
    x_s.append(x)
    y_sum = sum([val for _, val in data]) * 1.0
    s = 0
    upper_y, lower_y = [], []
    for _, val in data:
        s += val
        y_val = s / y_sum
        #y_u = min(y_val + ks_d_val, 1)
        #y_l = max(y_val - ks_d_val, 0)
        y_u = min(y_val + d_val, 1)
        y_l = max(y_val - d_val, 0)
        upper_y.append(y_u)
        lower_y.append(y_l)
    y_s.append(lower_y)
    y_s.append(upper_y)

    full_path = '%s/%d.png' % (path, day_index)
    plt.plot(x_s[0], y_s[0], 'r-', label='Diagnosed')
    plt.plot(x_s[1], y_s[1], 'b--', label='Control')
    plt.plot(x_s[2], y_s[2], 'k-.', label='Lower confidence')
    plt.plot(x_s[3], y_s[3], 'k-.', label='Upper confidence')
    plt.xlabel("Normalized fractional ranking")
    plt.ylabel("Percentage of population")
    plt.legend()
    plt.title(plot_info["title"] + "\nDistribution on day %d from diagnosis" %
              (day_index - len(ONSET_PERIOD) / 2))
    plt.savefig(full_path)
    plt.clf()
def plot_ranked_line_distribution(sick_counters, healthy_counters, onset_date_base_comp_ids, plot_info):
    sick_counters = np.array(sick_counters).T
    healthy_counters = np.array(healthy_counters).T
    onset_date, _, _ = onset_date_base_comp_ids[0]

    def remove_missing_data(counters):
        new_ranks = []
        for uid_ranks in counters:
            new_ranks.append([v for v in uid_ranks if not np.isnan(v)])
        return new_ranks

    sick_counters = remove_missing_data(sick_counters)
    healthy_counters = remove_missing_data(healthy_counters)

    _, ks_d_values = significance_ks_test(sick_counters, healthy_counters)

    print("Creating distribution plots")
    assert len(sick_counters) == len(healthy_counters)
    #create_distribution_figure_multiple_days(plot_info, onset_date, sick_counters, healthy_counters, ks_d_values)
    create_distribution_plots_shared_y_axis(plot_info, onset_date, sick_counters, healthy_counters, ks_d_values)

    #for day_index in tqdm(range(len(sick_counters))):
    #    create_distribution_figure_parallel(plot_info, onset_date, day_index, sick_counters, healthy_counters, ks_d_values)
def plot_ranked_line(sick_counters, healthy_counters, num_onset_dates, num_sick_users, num_healthy_users, onset_date, plot_info):
    sick_counters_transposed  = list(np.array(sick_counters).T)
    healthy_counters_transposed = list(np.array(healthy_counters).T)
    lines = PARALLEL(
        delayed(bootstrap_mean_confidence_interval)(day_counter) 
        for day_counter in tqdm(sick_counters_transposed + healthy_counters_transposed, desc="Bootstrapping")
    )
    sick_line = np.array(lines[:len(sick_counters_transposed)]).T
    healthy_line = np.array(lines[len(healthy_counters_transposed):]).T

    try:
        if plot_info['time_selected'] == TimeSpanSelected.LongTerm:
            assert len(LONG_TERM_DATE_RANGE) == len(sick_line[0]) == len(healthy_line[0])
        else:
            assert len(ONSET_PERIOD) == len(sick_line[0]) == len(healthy_line[0])
    except:
        print("!ERROR CAUGHT!")
        print("Insufficient data when generating")
        print(plot_info['plot_name'], plot_info['time_selected'])
        print("Lengths of ONSET_PERIOD, sick_line and healthy_line")
        print(len(ONSET_PERIOD), len(sick_line[0]), len(healthy_line[0]))
        return
    
    try:
        #mw_p_values = significance_mann_whitney_u_test(sick_counters, healthy_counters)
        #significance_fdr_correction(plot_info, onset_date, mw_p_values, sick_line, healthy_line, "MW")

        ks_p_values, ks_d_values = significance_ks_test(sick_counters, healthy_counters)
        significance_fdr_correction(plot_info, onset_date, ks_p_values, ks_d_values, sick_line, healthy_line, "KS")
    except Exception as ex:
        print("SIGNIFICANCE TESTS FAILED!")
        print(ex)


    fig = plot_ranked_line_layout(sick_line, healthy_line, plot_info,
                                  num_onset_dates, num_sick_users, num_healthy_users, onset_date)
    save_figure(fig, plot_info)

# =====================================
#            Plot Long Term
# =====================================

def plot_long_term(sick_uids, healthy_uids, start_date, end_date, is_nan_feature, plot_info, calc_callback):
    bin_id = plot_info["bin_id"]
    if PLOT_TYPE == PlotType.RankedLine:
        print("Creating LongTerm RankedLine graph")
        past_day_count = 10
        future_day_count = 0
        sick_ranks = get_individual_ranks(sick_uids, start_date, end_date, bin_id, past_day_count, future_day_count, is_nan_feature, calc_callback)
        healthy_ranks = get_individual_ranks(healthy_uids, start_date, end_date, bin_id, past_day_count, future_day_count, is_nan_feature, calc_callback)

        plot_ranked_line(sick_ranks, healthy_ranks, None, len(sick_uids), len(healthy_uids), (start_date, end_date), plot_info)
    else:
        print("Creating LongTerm Classic graph")
        plot_classic_for_long_term(sick_uids, start_date, end_date, plot_info, is_nan_feature, calc_callback)

# =====================================
#           Feature Plotters
# =====================================

def plot_week(onset_and_uids, time_selected, hours_per_bin, is_nan_feature, plot_info, *calc_callback):
    for bin_id in range(max(1, 24 // hours_per_bin)):
        plot_info["bin_id"] = bin_id
        plot_info["hours_per_bin"] = hours_per_bin
        plot_info["time_selected"] = time_selected
        # print(onset_and_uids)
        plot_info["file_names"] = file_names = plot_names(plot_info, onset_and_uids[0][0])
        if not should_regenerate_file(file_names):
            print("Skipping: ", file_names[0])
            return
        print("Creating: ", file_names[0])

        if time_selected == TimeSpanSelected.LongTerm: 
            if FILE_IN_USE == "large":
                sick_uids, _ = np.loadtxt(CONTROL_GROUP_FILE, delimiter=',', unpack=True)
                sick_uid_set = set(sick_uids)
                all_uid_set = set(user_day_feature_dict.keys())
                sick_uids = list(all_uid_set & sick_uid_set)
                healthy_uids = list(all_uid_set - sick_uid_set)
            else:
                sick_uids = user_day_feature_dict.keys()
                healthy_uids = sick_uids[:10]
            onset_date_base_comp_ids
            plot_long_term(sick_uids, healthy_uids, LONG_TERM_START_DATE, LONG_TERM_END_DATE, is_nan_feature, plot_info, calc_callback)
            return

        if PLOT_TYPE == PlotType.Classic:
            plot_classic_for_onsets(onset_and_uids, is_nan_feature, plot_info, calc_callback)
        elif PLOT_TYPE == PlotType.RankedLine:
            sick_counters, healthy_counters = get_ranked_data_for_onsets(
                onset_date_base_comp_ids, plot_info, is_nan_feature, calc_callback)

            num_onset_dates = len([onset_date for onset_date, _, _ in onset_date_base_comp_ids])
            num_sick_users = len([uid for _, sick_uids, _ in onset_date_base_comp_ids for uid in sick_uids])
            num_healthy_users = len([uid for _, _, healthy_uids in onset_date_base_comp_ids for uid in healthy_uids])
            onset_date, _, _ = onset_date_base_comp_ids[0]
            #plot_ranked_line(sick_counters, healthy_counters, num_onset_dates, num_sick_users, num_healthy_users, onset_date, plot_info)
            plot_ranked_line_distribution(sick_counters, healthy_counters, onset_date_base_comp_ids, plot_info)

# =====================================
#              Read Data
# =====================================

def uids_not_diagnosed_near_onset_date(onset_date):
    exclude_period = [
        onset_date + dt.timedelta(days) 
        for days in EXCLUSION_PERIOD_FOR_HEALTHY_USERS
    ]
    return [uid 
        for onset, uids in day_uids_onset
        for uid in uids 
        if onset not in exclude_period
    ]
def read_data(collect_loc=True, collect_call_text=True):
    global day_uids_onset, binned_user_day_feature_dicts, onset_date_base_comp_ids

    data_reader = DataReader(
        HOURS_PER_BIN_LIST,
        ONSET_AND_CONTROL_GROUP_MAPPING_FILE,
        extra_location_file=EXTRA_LOCATION_DATA_FILE
    )
    print("Reading data")
    binned_user_day_feature_dicts = data_reader.read_call_data({}, CALL_DATA_FILE, collect_loc=collect_loc, collect_call_text=collect_call_text)
    all_uids = set(binned_user_day_feature_dicts[HOURS_PER_BIN_LIST[0]].keys())
    #data_reader.read_extra_loc_data(binned_user_day_feature_dicts, uids_to_use=all_uids)

    print("Reading Onset data")
    onset_dicts = data_reader.read_onset_and_control_group_mapping_file(all_uids)
    day_uids_onset = onset_dicts[0].items()
    onset_date_base_comp_ids = [(onset, sick, uids_not_diagnosed_near_onset_date(onset)) for onset, sick in day_uids_onset]

def read_data_control_groups(collect_loc=True, collect_call_text=True):
    global day_uids_onset, binned_user_day_feature_dicts, onset_date_base_comp_ids
    data_reader = DataReader(
        HOURS_PER_BIN_LIST,
        ONSET_AND_CONTROL_GROUP_MAPPING_FILE,
        extra_location_file=EXTRA_LOCATION_DATA_FILE
    )

    print("Reading Onset data")
    onset_dicts = data_reader.read_onset_and_control_group_mapping_file()
    day_uids_onset = onset_dicts[0].items()
    day_uids_onset_control = onset_dicts[1:][CONTROL_GROUP_TO_USE]
    onset_date_base_comp_ids = [(onset, sick, day_uids_onset_control[onset]) for onset, sick in day_uids_onset]
    all_uids = set([uid 
        for dict in [onset_dicts[0], day_uids_onset_control]
        for uids in dict.values() 
        for uid in uids
    ])

    print("Reading G1 data")
    binned_user_day_feature_dicts = {}
    data_reader.read_call_data(binned_user_day_feature_dicts, G1_CONTROL_FILE, uids_to_use=all_uids, collect_loc=collect_loc, collect_call_text=collect_call_text)
    data_reader.read_call_data(binned_user_day_feature_dicts, G1_DIAGNOSED_FILE, uids_to_use=all_uids, collect_loc=collect_loc, collect_call_text=collect_call_text)
    data_reader.read_extra_loc_data(binned_user_day_feature_dicts, uids_to_use=all_uids)

def read_data_long_term(collect_loc=True, collect_call_text=True):
    global day_uids_onset, binned_user_day_feature_dicts
    data_reader = DataReader(
        HOURS_PER_BIN_LIST,
        ONSET_AND_CONTROL_GROUP_MAPPING_FILE,
        G1_DATA_FILE if FILE_IN_USE == 'large' else CALL_DATA_FILE,
        extra_location_file=EXTRA_LOCATION_DATA_FILE
    )
    before = [LONG_TERM_START_DATE - dt.timedelta(i) for i in range(70)]
    after = [LONG_TERM_END_DATE + dt.timedelta(i) for i in range(70)]
    if PLOT_TYPE == PlotType.Classic:
        collection_period = before + LONG_TERM_DATE_RANGE + after
    else:
        collection_period = before + LONG_TERM_DATE_RANGE 

    sick_uids, control_uids = np.loadtxt(CONTROL_GROUP_FILE, delimiter=',', unpack=True)
    day_uids_onset = data_reader.read_onset_data(sick_uids).items()
    binned_user_day_feature_dicts = data_reader.read_call_data(use_date_period=set(collection_period), collect_loc=collect_loc, collect_call_text=collect_call_text)    
    data_reader.read_extra_loc_data(binned_user_day_feature_dicts, uids_to_use=sick_uids + control_uids)

# =====================================
#        Export data to h5
# =====================================
def append_to_dataset(inputs_set, write_array):
    arr_len =  len(write_array)
    inputs_set.resize(inputs_set.shape[0] + arr_len, axis=0)
    try:
        inputs_set[-arr_len:] = write_array
    except Exception as ex:
        print("ERROR caught!", ex)
        traceback.print_exc()
        traceback.print_stack()
        exit()
        #print(len(write_array), write_array)
        #print(inputs_set.shape, inputs_set)

def process_data_and_create_h5(uids_and_days, num_days, filename, plotters):
    global user_day_feature_dict
    feature_count = len(plotters)
    for hour_bin_size in HOURS_PER_BIN_LIST:
        bin_count = max(1, 24 // hour_bin_size)
        row_len = num_days * feature_count * bin_count
        file_path = H5_DATA_PATH % (hour_bin_size, filename) 
        file_name = H5_DATA_FILE % (feature_count, num_days)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        print("Creating: %s" % (file_path + file_name))
        with h5py.File(file_path + file_name, 'w') as h5f:
            inputs_set = h5f.create_dataset("inputs", (0, row_len), maxshape=(None, row_len))
            user_id_set = h5f.create_dataset("user_ids", (0, ), maxshape=(None, ))
            label_set = h5f.create_dataset("labels", (0, num_days), maxshape=(None, num_days))
            h5f.create_dataset("bin_count", data=[bin_count])
            
            feature_names = [p()[0]['plot_name'] for p in plotters]
            h5f.create_dataset("feature_names", data=feature_names)

            user_day_feature_dict = binned_user_day_feature_dicts[hour_bin_size]

            all_days = []
            for uid, days, onset_day in tqdm(uids_and_days):
                day_list = []
                for today in days:
                    feature_list = []
                    for bin_id in range(bin_count):
                        for plotter in plotters:
                            params = plotter()
                            [[calculated_value]] = get_individual_ranks([uid], today, today, bin_id, 10, 0, is_nan_plotter(plotter), params[1:], progress_bar=no_progress_bar, do_parallel=False)
                            feature_list.append(calculated_value)
                    imputed_arr = []
                    try:
                        imputer = Imputer(missing_values='NaN', strategy='median')
                        imputed_arr = imputer.fit_transform(feature_list)
                    except Exception as ex:
                        pass
                    if len(imputed_arr) > 0:
                        day_list.extend(imputed_arr)
                    else:
                        day_list.extend(feature_list)
                all_days.append(day_list)
                append_to_dataset(user_id_set, [uid])
                labels = [[(1 if day == onset_day else 0) for day in days]]
                append_to_dataset(label_set, labels)
            imputer = Imputer(missing_values='NaN', strategy='median')
            all_days = imputer.fit_transform(all_days)
            append_to_dataset(inputs_set, all_days)

def export_relative_dates(uids_and_onsets, filename, plotters):
    days_before_onset = 20
    days_after_onset = 20
    uids_and_days = []
    for onset_day, uids in uids_and_onsets:
        for uid in uids:
            uids_and_days.append((
                uid,
                [onset_day + dt.timedelta(x) for x in range(-days_before_onset, days_after_onset+1)],
                onset_day
                ))
    process_data_and_create_h5(uids_and_days, len(range(-days_before_onset, days_after_onset+1)), filename, plotters)

def export_absolute_dates(uids_and_onsets, filename, plotters):
    start_date = dt.date(2009, 6, 4)
    days_to_use_for_all = [start_date + dt.timedelta(x) for x in range(250)]
    print(days_to_use_for_all[0], days_to_use_for_all[-1])
    uids_and_days = []
    for onset_day, uids in uids_and_onsets:
        for uid in uids:
            uids_and_days.append((
                uid,
                days_to_use_for_all,
                onset_day
                ))
    process_data_and_create_h5(uids_and_days, len(days_to_use_for_all), filename, plotters)

def export_machine_learning_data(uids_and_onsets, sick_or_healthy_str, plotters):
    uids_to_use = [uid for _, uids in uids_and_onsets for uid in uids]
    random.seed(1337)
    random.shuffle(uids_to_use)

    val_ratio = 0.2
    val_size = int(len(uids_to_use) * val_ratio)
    val_users = uids_to_use[:val_size]
    train_and_test_users = uids_to_use[val_size:]
    train_ratio = 0.7
    train_size = int(len(train_and_test_users) * train_ratio)
    train_users = train_and_test_users[:train_size]
    test_users = train_and_test_users[train_size:]

    uids_and_onsets_train = []
    uids_and_onsets_test = []
    uids_and_onsets_val = []
    for onset, uids in uids_and_onsets:
        train_filtered = []
        test_filtered = []
        val_filtered = []
        for uid in uids:
            if uid in train_users:
                train_filtered.append(uid)
            if uid in test_users:
                test_filtered.append(uid)
            if uid in val_users:
                val_filtered.append(uid)
        if len(train_filtered) > 0:
            uids_and_onsets_train.append((onset, train_filtered))
        if len(test_filtered) > 0:
            uids_and_onsets_test.append((onset, test_filtered))
        if len(val_filtered) > 0:
            uids_and_onsets_val.append((onset, val_filtered))

    print("Creating h5 train file")
    #export_relative_dates(uids_and_onsets_train, "relative_days_%s_train" % sick_or_healthy_str, plotters)
    export_absolute_dates(uids_and_onsets_train, "absolute/%s_train" % sick_or_healthy_str, plotters)
    print("Creating h5 test file")
    #export_relative_dates(uids_and_onsets_test, "relative_days_%s_test" % sick_or_healthy_str, plotters)
    export_absolute_dates(uids_and_onsets_test, "absolute/%s_test" % sick_or_healthy_str, plotters)
    print("Creating h5 val file")
    #export_relative_dates(uids_and_onsets_val, "relative_days_%s_val" % sick_or_healthy_str, plotters)
    export_absolute_dates(uids_and_onsets_val, "absolute/%s_val" % sick_or_healthy_str, plotters)

# =====================================
#           Plot Stats
# =====================================

HEALTHY_EXPORT_STATS_FILE = "healthy_feature_stats_table.tex"
SICK_EXPORT_STATS_FILE = "sick_feature_stats_table.tex"
def export_plot_header(file):
    with open(file, 'w') as tex_file:
        tex_file.write("\\begin{tabular}{c|c|c|c|c|c}\n")
        tex_file.write("Feature & Min & Max & Mean & Median & Mode \\\\\n")
        tex_file.write("\\hline\n")

def export_plot_stats(plot_info, uid_day_stats, file):
    raw_array = np.array(uid_day_stats)

    mean_per_uid = np.nanmean(raw_array, axis=1) 
    mean = np.nanmean(mean_per_uid)

    median_per_uid = np.nanmedian(raw_array, axis=1) 
    median = np.nanmedian(median_per_uid)

    mode_per_uid, _ = sp.stats.mode(raw_array, axis=1, nan_policy='omit') 
    mode, _ = sp.stats.mode(mode_per_uid, nan_policy='omit')
    
    with open(file, 'a') as tex_file:
        tex_file.write("%s & %.3f & %.3f & %.3f & %.3f & %.3f \\\\\n" % (
            plot_info["plot_name"],
            np.nanmin(raw_array),
            np.nanmax(raw_array),
            mean,
            median,
            mode
        ))
def export_raw_distrbution(plot_info, healthy_data, sick_data):
    raw_healthy = np.array(healthy_data).flatten()
    raw_sick = np.array(sick_data).flatten()

    def get_trace(data, name):
        return { 
            "x": data,
            "name": name, 
            "type": "histogram", 
        }

    data = [
        get_trace(raw_healthy, "Control"),
        get_trace(raw_sick, "Diagnosed")
    ]
    layout = {
        "autosize": True, 
        "legend": {
            "x": 0.85, 
            "y": 1,
        }, 
        "width": 500,
        "height": 300,
        "margin": { "r": 30, "t": 60, "b": 60, "l": 60, },
        "barmode": "group", 
        "title": plot_info["title"], 
        "xaxis": {
            "title": "Extracted feature value", 
        }, 
        "yaxis": {
            "title": "Number of occurrences"
        }
    }
    fig = go.Figure(data=data, layout=layout)

    file_name = "histograms/raw_value_histogram_%s.pdf" % plot_info["plot_name"]
    plot_url = py.plot(fig, filename=file_name)
    py.image.save_as(fig, filename=file_name)

# =====================================
#           Main Function
# =====================================

def is_nan_plotter(plotter):
    return (plotter == nan_count_location or
            plotter == nan_count_call_in or 
            plotter == nan_count_call_out or 
            plotter == nan_count_text_in or 
            plotter == nan_count_text_out or
            plotter == unique_feature_with_nan_count) 

def run_and_catch(callback, plotters):
    try:
        callback(plotters)
        return True
    except pyex.PlotlyRequestError as ex:
        print("ERROR message: ")
        print(type(ex))
        print(ex)
        if "you've reached the threshold of" in ex:
            print("STOPPING execution because threshold has been reached!")
            exit()
        return False

def long_term(plotters):
    global user_day_feature_dict
    for hour_bin_size in HOURS_PER_BIN_LIST:
        user_day_feature_dict = binned_user_day_feature_dicts[hour_bin_size]
        for plotter in plotters:
            start = time.time()
            params = plotter()
            plot_week(day_uids_onset, TimeSpanSelected.LongTerm, hour_bin_size, is_nan_plotter(plotter), *params) 
            end = time.time()
            print("Time: ", end - start)
def all_days(plotters):
    global user_day_feature_dict, sick_uid_day_values_stats, healthy_uid_day_values_stats
    if SHOULD_COLLECT_STATS:
        export_plot_header(HEALTHY_EXPORT_STATS_FILE)
        export_plot_header(SICK_EXPORT_STATS_FILE)
    for hour_bin_size in HOURS_PER_BIN_LIST:
        user_day_feature_dict = binned_user_day_feature_dicts[hour_bin_size]
        for plotter in plotters:
            start = time.time()
            params = plotter()
            plot_week(day_uids_onset, TimeSpanSelected.AllDays, hour_bin_size, is_nan_plotter(plotter), *params)
            if SHOULD_COLLECT_STATS:
                plot_info = params[0]
                export_plot_stats(plot_info, healthy_uid_day_values_stats, HEALTHY_EXPORT_STATS_FILE)
                export_plot_stats(plot_info, sick_uid_day_values_stats, SICK_EXPORT_STATS_FILE)
                export_raw_distrbution(plot_info, healthy_uid_day_values_stats, sick_uid_day_values_stats)
                sick_uid_day_values_stats = []
                healthy_uid_day_values_stats = []
            end = time.time()
            print("Time: ", end - start)
def top_single_dates(plotters):
    global user_day_feature_dict
    for hour_bin_size in HOURS_PER_BIN_LIST:
        user_day_feature_dict = binned_user_day_feature_dicts[hour_bin_size]
        for plotter in plotters:
            for i in range(3):
                onset_and_uids = day_uids_onset[i]
                single_date = [onset_and_uids]
                params = plotter()
                plot_week(single_date, TimeSpanSelected.SingleDate, hour_bin_size, is_nan_plotter(plotter), *params) 
def all_weekdays(plotters):
    global user_day_feature_dict
    for hour_bin_size in HOURS_PER_BIN_LIST:
        user_day_feature_dict = binned_user_day_feature_dicts[hour_bin_size]
        for plotter in plotters:
            for i in range(7):
                all_groups_for_single_weekday = [
                    (onset, uids)
                    for onset, uids in day_uids_onset 
                    if onset.weekday() == i
                ]
                params = plotter()
                plot_week(all_groups_for_single_weekday, TimeSpanSelected.SingleDate, hour_bin_size, is_nan_plotter(plotter), *params) 

def thread_worker(plot_type_func, plotters):
    success = False
    while not success:
        success = run_and_catch(plot_type_func, plotters)

def main():
    plotters = [
        unique_locations,
        call_counts_outgoing,
        mean_call_dur_both,

        distance_traveled,
        new_locations_visited,
#
        nan_count_location,
        nan_count_call_in,
        nan_count_call_out,
        nan_count_text_in,
        nan_count_text_out,
#
        unique_contacts_outgoing,
        unique_contacts_incoming,
        unique_contacts_both,
        new_people_called_outgoing,
        new_people_called_incoming,
        new_people_called_both,
#
        duration_of_calls_incoming,
        duration_of_calls_outgoing,
        duration_of_calls_both,
#
        call_counts_incoming,
        #call_counts_outgoing,
        call_counts_both,
        text_counts_incoming,
        text_counts_outgoing,
        text_counts_both,
        call_and_text_count_incoming,
        call_and_text_count_outgoing,
        call_and_text_count_both,
#
        mean_call_dur_incoming,
        mean_call_dur_outgoing,
        #mean_call_dur_both,
#
        top_contacts_by_duration_incoming,
        top_contacts_by_duration_outgoing,
        top_contacts_by_duration_both,
        non_top_contacts_by_duration_incoming,
        non_top_contacts_by_duration_outgoing,
        non_top_contacts_by_duration_both,
#
        top_contacts_by_frequency_incoming,
        top_contacts_by_frequency_outgoing,
        top_contacts_by_frequency_both,
        non_top_contacts_by_frequency_incoming,
        non_top_contacts_by_frequency_outgoing,
        non_top_contacts_by_frequency_both,

        # Not converted to parallel yet
        # plot_intermediate_call_and_text_time,
    ]

    collect_loc, collect_call_text = True, True
    if EXPORT_TO_H5:
        if FILE_IN_USE == 'large':
            # Read data for sick and healthy groups
            read_data_control_groups(collect_loc, collect_call_text)
            sick_onsets = [(onset_day, sick_uids) for onset_day, sick_uids, _ in onset_date_base_comp_ids]
            healthy_onsets = [(onset_day, healthy_uids) for onset_day, _, healthy_uids in onset_date_base_comp_ids]
            export_machine_learning_data(sick_onsets, "sick", plotters)
            export_machine_learning_data(healthy_onsets, "healthy", plotters)
        else:
            read_data(collect_loc, collect_call_text)
            export_machine_learning_data(day_uids_onset, "sick", plotters)
        return
    if TimeSpanSelected.LongTerm in TIME_SPANS:
        read_data_long_term(collect_loc, collect_call_text)
    elif USE_CONTROL_GROUP:
        read_data_control_groups(collect_loc, collect_call_text)
    else:
        read_data(collect_loc, collect_call_text)

    print("signing in to plotly")
    # Replace the username, and API key with your credentials.
    py.sign_in('thorgeirk11', '??')  # Pro features
    # py.sign_in('sir.thorgeir', '??')
    # py.sign_in('asfalt', '??')

    for time_span in TIME_SPANS:
        if time_span == TimeSpanSelected.AllDays:
            thread_worker(all_days, plotters)
        if time_span == TimeSpanSelected.SingleDate:
            thread_worker(top_single_dates, plotters)
        if time_span == TimeSpanSelected.Weekday:
            thread_worker(all_weekdays, plotters)
        if time_span == TimeSpanSelected.LongTerm:
            thread_worker(long_term, plotters)
    
if __name__ == "__main__":
    main()

