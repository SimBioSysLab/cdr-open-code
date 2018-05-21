from multiprocessing import Queue, Process
import pandas as pd
import numpy as np
import ciso8601
import os
import plotly.exceptions as pyex
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import datetime as dt
import scipy as sp
import multiprocessing as mp
from scipy.stats.mstats import gmean
import scipy.stats
import math
import time
from math import sin, cos, sqrt, atan2, radians
from utils.CDR_csv_reader import DataReader


# Used for selecting healthy individuals for comparison in ranked line.
# Users may now be sick with this period to be selected for comparison. 
EXCLUSION_PERIOD_FOR_HEALTHY_USERS = range(-14, 15)

# Fiter out uids if they have too few records for the onset week. 
NUMBER_OF_ACTIVE_DAYS_IN_ONSET_WEEK = 3

# Only regenerate the graph if it is more than a week old
REFRESH_TIME_FOR_GRAPHS = 86400*7
# Macro used to read huge csv files in chunks
CHUNK_SIZE = 10**5
#CHUNK_SIZE = None

# File selection macros
ONSET_DATA_FILE = '../../inputData/FULLonset.csv'
LARGE_DATA_FILE = '../../inputData/FULLg1sickcalls.csv'
SMALL_DATA_FILE = '../../inputData/g1sickcalls.csv'

CALL_DATA_FILE, FILE_IN_USE = (LARGE_DATA_FILE, "large")
#CALL_DATA_FILE, FILE_IN_USE = (SMALL_DATA_FILE, "small")

# List of bin sizes to run
HOURS_PER_BIN_LIST = [24]

# One Week
BEFORE_ONSET_PERIOD = range(-73,-3)
ONSET_PERIOD = range(-3,4)
AFTER_ONSET_PERIOD = range(4,74)


# =====================================
#          Social Network
# =====================================
def prev_weekdays(current_day, day_dict, number_of_previous_days=10):
    return [ current_day + dt.timedelta(x)
        for x in (range(-number_of_previous_days * 7, 0, 7) if number_of_previous_days > 0 else range(1))
            if current_day + dt.timedelta(x) in day_dict
    ]

def prev_days(current_day, day_dict, number_of_previous_days=30):
    return [ current_day + dt.timedelta(x)
        for x in (range(-number_of_previous_days, 0) if number_of_previous_days > 0 else range(1))
            if current_day + dt.timedelta(x) in day_dict
    ]

def features_for_prev_days(current_day, day_dict, feature_list, number_of_previous_days):
    # Get days where data exists for this uid for the given hour bin id
    before = prev_days(current_day, day_dict, number_of_previous_days)
    prev_features = {}
    for feature_type in feature_list:
        if feature_type not in prev_features:
            prev_features[feature_type] = {}
        for past_day in before:
            for feature_dict in day_dict[past_day].values():
                mergin_feature = prev_features[feature_type]

                # Merge the feature dicts for past days into a single dict.
                # E.g. call_in feature will be single entry in the dict. 
                for obj_id, events in feature_dict[feature_type].items():
                    if obj_id not in mergin_feature:
                        mergin_feature[obj_id] = []
                    mergin_feature[obj_id].extend(events)
    return prev_features

def top_contacts_helper(dur_or_freq, current_day, day_dict, days_before_onset):
    top_contacts_dict = {}
    feature_type_list = ["call_in", "call_out"]
    past_feature_dict = features_for_prev_days(current_day, day_dict, feature_type_list, days_before_onset)
    # Populate the top contacts list based on previous days
    for feature_type in feature_type_list:
        top_contacts_feature_dict = {}
        for obj_id, events in past_feature_dict[feature_type].items():
            if obj_id not in top_contacts_feature_dict:
                top_contacts_feature_dict[obj_id] = 0
            if dur_or_freq == "frequency":
                top_contacts_feature_dict[obj_id] += len(events)
            elif dur_or_freq == "duration":
                for duration, timestamp in events:
                    top_contacts_feature_dict[obj_id] += duration
        top_contacts_dict[feature_type] = top_contacts_feature_dict

    merged_dict = {}
    for feature_type in feature_type_list:
        for uid, mesure in top_contacts_dict[feature_type].items():
            if uid in merged_dict:
                merged_dict[uid] += mesure
            else:
                merged_dict[uid] = mesure

    sorted_list = sorted(merged_dict.items(), key=lambda (k,v): v, reverse=True)
    return sorted_list

def top_contacts_by_call_duration(current_day, day_dict, days_before_onset=30):
    return top_contacts_helper("duration", current_day, day_dict, days_before_onset)

def top_n_contacts_by_call_frequency(current_day, day_dict, days_before_onset=30):
    return top_contacts_helper("frequency", current_day, day_dict, days_before_onset)

# =====================================
#           Main Function
# =====================================

data_reader = DataReader(
    CALL_DATA_FILE, 
    ONSET_DATA_FILE, 
    NUMBER_OF_ACTIVE_DAYS_IN_ONSET_WEEK, 
    HOURS_PER_BIN_LIST
)


print("Reading call data")
binned_user_day_feature_dicts = data_reader.read_call_data(None)

print("Reading Onset data")
day_uids_onset = data_reader.read_onset_data(binned_user_day_feature_dicts[HOURS_PER_BIN_LIST[0]])
print "Number of uids, before filtering", sum(len(uids) for _, uids in day_uids_onset.items())
day_uids_onset = data_reader.remove_uids_with_no_data_during_onset_week(ONSET_PERIOD, day_uids_onset, binned_user_day_feature_dicts[HOURS_PER_BIN_LIST[0]])
print "Number of uids, after filtering", sum(len(uids) for _, uids in day_uids_onset.items())
day_uids_onset = sorted(day_uids_onset.items(), key=lambda tup: len(tup[1]))[::-1]

print("signing in to plotly")
# Replace the username, and API key with your credentials.
py.sign_in('thorgeirk11', '??')  # Pro features
#py.sign_in('sir.thorgeir', '??')
#py.sign_in('asfalt', '??')

def get_strength_tie_for_day(day, uids, user_day_feature_dict, use_duration):
    data_points = {}
    for uid in uids:
        if day not in user_day_feature_dict[uid]:
            continue 
        day_dict = user_day_feature_dict[uid]
        if use_duration:
            tops = top_contacts_by_call_duration(day, day_dict, 30)
        else:
            tops = top_n_contacts_by_call_frequency(day, day_dict, 30)
        top_contacts = [contact for contact, _ in tops]

        if use_duration:
            today_tops = top_contacts_by_call_duration(day, day_dict, 0)
        else:
            today_tops = top_n_contacts_by_call_frequency(day, day_dict, 0)
        today_contacts = [contact for contact, _ in today_tops if contact in top_contacts]
        today_vals = [dur for contact, dur in today_tops if contact in top_contacts]

        sum_vals = sum(today_vals) * 1.0
        vals_list = [val / sum_vals for val in today_vals]

        top_contact_len = len(top_contacts) * 1.0
        index_list = [top_contacts.index(uid) / top_contact_len for uid in today_contacts]

        for i in range(len(index_list)):
            x_val = index_list[i]
            y_val = vals_list[i]
            if x_val in data_points:
                data_points[x_val] += y_val
            else:
                data_points[x_val] = y_val
    return data_points.items()

def create_CCDF_trace(data_points, name):
    sorted_list = sorted(data_points, key=lambda (k, v): k)
    x = [key for key, _ in sorted_list]
    y = [val for _, val in sorted_list]
    sum_y = sum(y)
    y = [val / sum_y for val in y]
    y = np.cumsum(y)
    y = [1 - val for val in y]
    return go.Scatter(
        name=name,
        x=x,
        y=y,
        mode='lines+markers',
        opacity=0.7,
        line={"shape":"spline"}
    )

def create_simple_line_trace(data_points, name):
    sorted_list = sorted(data_points, key=lambda (k, v): k)
    x = [key for key, _ in sorted_list]
    y = [val for _, val in sorted_list]
    sum_y = sum(y) * 1.0
    y = [val / sum_y for val in y]
    return go.Scatter(
        name=name,
        x=x,
        y=y,
        mode='lines+markers',
        opacity=0.7,
        line={"shape":"spline"}
    )

def strength_tie_single_dates(use_duration):
    onset_sick_and_healthy_uids = []

    dur_fre_text = ("Duration" if use_duration else "Frequency")

    for onset_date, sick_uids in day_uids_onset:
        exclude_period = [onset_date + dt.timedelta(days) for days in EXCLUSION_PERIOD_FOR_HEALTHY_USERS]
        healthy_uids = [uid for onset, uids in day_uids_onset for uid in uids if onset not in exclude_period]
        onset_sick_and_healthy_uids.append((onset_date, sick_uids, healthy_uids))
    
    def get_layout(y_max, use_log_scale, is_cum_text):
        return go.Layout(
            bargap=0.1,
            bargroupgap=0,
            barmode="group",
            title='Tie strength based on %s' % (dur_fre_text.lower()),
            width=800, height=640,
            xaxis=dict(
                title='Percentile rank of contact',
                type='log' if use_log_scale else 'line',
                autorange=use_log_scale,
                range=[0, 1]
            ),
            yaxis=dict(
                title='%s of calls %s' % (dur_fre_text, is_cum_text),
                type='log' if use_log_scale else 'line',
                autorange=use_log_scale,
                range=[0, y_max]
            ),
            legend={ "orientation": "h" }
        )
    
    sick_data = []
    health_data = []
    for hour_bin_size in HOURS_PER_BIN_LIST:
        user_day_feature_dict = binned_user_day_feature_dicts[hour_bin_size]
        for onset_date, sick_uids, healthy_uids in onset_sick_and_healthy_uids:
            print onset_date, len(sick_uids), len(healthy_uids)
            for day in [onset_date + dt.timedelta(x) for x in range(3)]:
                sick_data.extend(get_strength_tie_for_day(onset_date, sick_uids, user_day_feature_dict, use_duration ))
                health_data.extend(get_strength_tie_for_day(onset_date, healthy_uids, user_day_feature_dict, use_duration))

    def unify(data):
        data_points = {}
        for key, val in data:
            new_key = round(key, 1)
            if new_key in data_points:
                data_points[new_key] += val
            else:
                data_points[new_key] = val
        return data_points.items()
    
    sick_data = unify(sick_data)
    health_data = unify(health_data)
    
    for use_cumulative in [False]:
        if use_cumulative:
            sick_trace = create_CCDF_trace(sick_data, "Sick")
            healthy_trace = create_CCDF_trace(health_data, "Healthy")
        else:
            sick_trace = create_simple_line_trace(sick_data, "Sick")
            healthy_trace = create_simple_line_trace(health_data, "Healthy")
            
        print sick_trace
        max_y = max(sick_trace.y) * 1.1
        type_text = ("(cumulative)" if use_cumulative else "")
        for use_log_scale in [False]:
            fig = go.Figure(data=[sick_trace, healthy_trace], layout=get_layout(max_y, use_log_scale, type_text))
            plot_url = py.plot(fig, filename="Graph")
            print plot_url

            #py.image.save_as(fig, filename='tie/tie_strength%s_%s_%s_%s.png' 
            #% ("_log" if use_log_scale else "", dur_fre_text, type_text, FILE_IN_USE))

def main():
    #strength_tie_single_dates(True)
    strength_tie_single_dates(False)
    

if __name__ == "__main__":
    main()

