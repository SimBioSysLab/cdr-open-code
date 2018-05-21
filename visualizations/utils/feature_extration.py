import pandas as pd
import numpy as np
import datetime as dt
from collections import Counter
from utils.plot_utils import plot_utils
import operator
import ciso8601
import math

#-------------------------

def get_in_out_str(features):
    in_out = list(set([feature[5:] for feature in features]))
    if len(in_out) == 1:
        in_or_out_str = "incoming" if in_out[0] == "in" else "outgoing"
        in_or_out_path_str = in_or_out_str
    else:
        in_or_out_str = "incoming and outgoing"
        in_or_out_path_str = "both"
    return (in_or_out_str, in_or_out_path_str)
def get_text_call_str(features):
    text_calls = list(set([feature[:4] for feature in features]))
    if len(text_calls) == 1:
        return text_calls[0] + "s"
    else:
        return "calls and texts"

# =====================================
#           Top Contacts
# =====================================

def top_contacts_calc(uid, day, bin_id, user_day_feature_dict, features, use_top, use_duration):
    not_top = 0
    top = 0
    for feature in features:
        day_dict = user_day_feature_dict[uid]
        feature_dict = day_dict[day][bin_id]
        if feature not in feature_dict:
            continue
        if use_duration:
            top_contacts = plot_utils.top_n_contacts_by_call_duration(False, day, day_dict)
        else:
            top_contacts = plot_utils.top_n_contacts_by_call_frequency(False, day, day_dict)
        for obj_id, call_to_object in feature_dict[feature].items():
            if obj_id not in top_contacts:
                not_top += 1
            else:
                top += 1
    return  (top if use_top else not_top)
def top_contacts(features, in_or_out, use_top, use_duration):
    use_top_str = "top" if use_top else "nontop"
    use_duration_str = "duration" if use_duration else "frequency"
    plot_info = {
        "title": "Calls to %s contacts by %s (%s)" % (use_top_str, use_duration_str, in_or_out),
        "yaxis": "Calls to %s contacts" % use_top_str,
        "plot_name": "%s_contacts_by_%s_%s" % (
            use_top_str.replace(' ', '_'),
            use_duration_str,
            in_or_out if len(features) == 1 else "both"
        ),
    }
    return plot_info, top_contacts_calc, features, use_top, use_duration

def top_contacts_by_duration_incoming():
    return top_contacts(['call_in'], 'incoming', True, True)
def top_contacts_by_duration_outgoing():
    return top_contacts(['call_out'], 'outgoing', True, True)
def top_contacts_by_duration_both():
    return top_contacts(['call_in', 'call_out'], 'incoming and outgoing', True, True)
def non_top_contacts_by_duration_incoming():
    return top_contacts(['call_in'], 'incoming', False, True)
def non_top_contacts_by_duration_outgoing():
    return top_contacts(['call_out'], 'outgoing', False, True)
def non_top_contacts_by_duration_both():
    return top_contacts(['call_in', 'call_out'], 'incoming and outgoing', False, True)

def top_contacts_by_frequency_incoming():
    return top_contacts(['call_in'], 'incoming', True, False)
def top_contacts_by_frequency_outgoing():
    return top_contacts(['call_out'], 'outgoing', True, False)
def top_contacts_by_frequency_both():
    return top_contacts(['call_in', 'call_out'], 'incoming and outgoing', True, False)
def non_top_contacts_by_frequency_incoming():
    return top_contacts(['call_in'], 'incoming', False, False)
def non_top_contacts_by_frequency_outgoing():
    return top_contacts(['call_out'], 'outgoing', False, False)
def non_top_contacts_by_frequency_both():
    return top_contacts(['call_in', 'call_out'], 'incoming and outgoing', False, False)

# =====================================
#           Count Features
# =====================================

def count_features_calc(uid, day, bin_id, user_day_feature_dict, features):
    count = 0
    day_dict = user_day_feature_dict[uid][day][bin_id]
    for feature in features:
        for call_to_object in day_dict.get(feature, {}).values():
            count += len(call_to_object)
    return count
def count_features(features):
    in_or_out_str, in_or_out_path_str = get_in_out_str(features)
    call_or_text_str = get_text_call_str(features)
    plot_info = {
        "title": "Number of %s (%s)" % (call_or_text_str, in_or_out_str),
        "yaxis": "Number of %s" % call_or_text_str,
        "plot_name": "%s_count_%s" % (
            call_or_text_str.replace(' ', '_'),
            in_or_out_path_str
        ),
    }
    return plot_info, count_features_calc, features

def call_counts_incoming():
    return count_features(['call_in'])
def call_counts_outgoing():
    return count_features(['call_out'])
def call_counts_both():
    return count_features(['call_out', 'call_in'])
def text_counts_incoming():
    return count_features(['text_in'])
def text_counts_outgoing():
    return count_features(['text_out'])
def text_counts_both():
    return count_features(['text_in', 'text_out'])
def call_and_text_count_incoming():
    return count_features(['text_in', 'call_in'])
def call_and_text_count_outgoing():
    return count_features(['text_out', 'call_out'])
def call_and_text_count_both():
    return count_features(['text_in', 'call_in','text_out', 'call_out'])

# =====================================
#           Mean Duration
# =====================================

def mean_call_dur_calc(uid, day, bin_id, user_day_feature_dict, features):
    day_dict = user_day_feature_dict[uid][day][bin_id]
    total_dur, call_count = 0.0, 0.0
    for feature in features:
        today_calls = day_dict.get(feature, {}).values()
        dur_of_calls = [dur for tuples in today_calls for dur, _ in tuples]
        total_dur += sum(dur_of_calls)
        call_count += len(dur_of_calls)
    if call_count > 0:
        return total_dur / call_count
    else:
        return 0
def mean_call_dur(features):
    in_or_out_str, in_or_out_path_str = get_in_out_str(features)
    plot_info = {
        "title": "Mean call duration (%s)" % in_or_out_str,
        "yaxis": "Mean call duration (sec)",
        "plot_name": "mean_call_duration_%s" % in_or_out_path_str
    }
    return plot_info, mean_call_dur_calc, features
def mean_call_dur_incoming():
    return mean_call_dur(['call_in'])
def mean_call_dur_outgoing():
    return mean_call_dur(['call_out'])
def mean_call_dur_both():
    return mean_call_dur(['call_in', 'call_out'])

# =====================================
#           Call Duration
# =====================================

def call_dur_calc(uid, day, bin_id, user_day_feature_dict, features):
    day_dict = user_day_feature_dict[uid][day][bin_id]
    total_dur = 0.0
    for feature in features:
        today_calls = day_dict.get(feature, {}).values()
        dur_of_calls = [dur for tuples in today_calls for dur, _ in tuples]
        total_dur += sum(dur_of_calls)
    return total_dur
def call_dur(features):
    in_or_out_str, in_or_out_path_str = get_in_out_str(features)
    plot_info = {
        "title": "Call duration (%s)" % in_or_out_str,
        "yaxis": "Call duration (sec)",
        "plot_name": "call_duration_%s" % in_or_out_path_str
    }
    return plot_info, call_dur_calc, features

def duration_of_calls_incoming():
    return call_dur(['call_in'])
def duration_of_calls_outgoing():
    return call_dur(['call_out'])
def duration_of_calls_both():
    return call_dur(['call_in', 'call_out'])

# =====================================
#           Unique Contacts
# =====================================

def unique_contacts_calc(uid, day, bin_id, user_day_feature_dict, features):
    day_dict = user_day_feature_dict[uid][day][bin_id]
    contacts = set()
    for feature in features:
        contacts.update(day_dict.get(feature, {}).keys())
    return len(contacts)
def unique_contacts(features):
    in_or_out_str, in_or_out_path_str = get_in_out_str(features)
    plot_info = {
        "title": "Unique contacts (calls %s)" % in_or_out_str,
        "yaxis": "Unique contacts",
        "plot_name": "unique_contact_count_%s" % in_or_out_path_str
    }
    return plot_info, unique_contacts_calc, features

def unique_contacts_incoming():
    return unique_contacts(['call_in'])
def unique_contacts_outgoing():
    return unique_contacts(['call_out'])
def unique_contacts_both():
    return unique_contacts(['call_in', 'call_out'])

def new_contacts_calc(uid, day, bin_id, user_day_feature_dict, features):
    day_dict = user_day_feature_dict[uid]
    new_contacts = set()
    for feature in features:
        new_contacts.update(plot_utils.unseen_features_for_prev_days(day_dict, feature, day, bin_id))
    return len(new_contacts)
def new_contacts(features):
    in_or_out_str, in_or_out_path_str = get_in_out_str(features)
    plot_info = {
        "title": "New contacts (%s)" % in_or_out_str,
        "yaxis": "New contacts called",
        "plot_name": "new_contacts_%s" % in_or_out_path_str,
    }
    return plot_info, new_contacts_calc, features

def new_people_called_outgoing():
    return new_contacts(['call_out'])
def new_people_called_incoming():
    return new_contacts(['call_in'])
def new_people_called_both():
    return new_contacts(['call_out','call_in'])

# =====================================
#           NaN Count
# =====================================

def unique_feature_with_nan_count_calc(uid, day, bin_id, user_day_feature_dict, feature):
    dict = user_day_feature_dict[uid]
    if day not in dict:
        return 0
    dict = dict[day]
    if bin_id not in dict:
        return 0
    dict = dict[bin_id]
    if feature not in dict:
        return 0
    features = dict[feature]
    return len(features)

def nan_count_calc(uid, day, bin_id, user_day_feature_dict, feature):
    dict = user_day_feature_dict[uid]
    if day not in dict:
        return 1
    dict = dict[day]
    if bin_id not in dict:
        return 1
    dict = dict[bin_id]
    if feature not in dict:
        return 1
    features = dict[feature]
    return 0 if len(features) > 0 else 1

def unique_feature_with_nan_count():
    plot_info = {
        "title": "Unique locations visited (including 0)",
        "yaxis": "Nr. of people with no location data",
        "plot_name": "unique_loc_visited_with_nan_count",
    }
    return plot_info, unique_feature_with_nan_count_calc, "location"
def nan_count_location():
    plot_info = {
        "title": "Individuals with no location data",
        "yaxis": "Nr. of people with no location data",
        "plot_name": "nan_count_location",
    }
    return plot_info, nan_count_calc, "location"
def nan_count_call_in():
    plot_info = {
        "title": "Individuals with no incoming calls",
        "yaxis": "Nr. of people with no incoming calls",
        "plot_name": "nan_count_call_in",
    }
    return plot_info, nan_count_calc, "call_in"
def nan_count_call_out():
    plot_info = {
        "title": "Individuals with no outgoing calls",
        "yaxis": "Nr. of people with no outgoing calls",
        "plot_name": "nan_count_call_out",
    }
    return plot_info, nan_count_calc, "call_out"
def nan_count_text_in():
    plot_info = {
        "title": "Individuals with no incoming texts",
        "yaxis": "Nr. of people with no incoming texts",
        "plot_name": "nan_count_text_in",
    }
    return plot_info, nan_count_calc, "text_in"
def nan_count_text_out():
    plot_info = {
        "title": "Individuals with no outgoing texts",
        "yaxis": "Nr. of people with no outgoing texts",
        "plot_name": "nan_count_text_out",
    }
    return plot_info, nan_count_calc, "text_out"

# =====================================
#           Location
# =====================================

def unique_locations_calc(uid, day, bin_id, user_day_feature_dict):
    return len(user_day_feature_dict[uid][day][bin_id].get("location", {}))
def unique_locations():
    plot_info = {
        "title": "Unique locations visited",
        "yaxis": "Unique locations visited",
        "plot_name": "unique_locations_visited",
    }
    return plot_info, unique_locations_calc

def position_distance(now, prev):
    try:
        # Approximate earth radius in Iceland
        earth_radius = 6360.5
        lat1, lon1 = now
        lat2, lon2 = prev
        c = math.pi / 180.0
        lat1 *= c
        lat2 *= c
        lon1 *= c
        lon2 *= c
        a = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * \
            math.cos(lat2) * math.cos(lon2 - lon1)
        return math.acos(a) * earth_radius
    except ValueError:
        return 0
def distance_traveled_calc(uid, day, bin_id, user_day_feature_dict):
    locations = user_day_feature_dict[uid][day][bin_id].get("location", {})

    loc_timstamp_list = [(loc, timestamp) 
        for loc, timestamps in locations.items() 
        for timestamp in timestamps
    ]
    sorted_loc_timestamps = sorted(loc_timstamp_list, key=lambda tup: tup[1])
    loc_in_order = [loc for loc, _ in sorted_loc_timestamps]

    number_of_loc = len(loc_in_order) - 1
    distance = 0
    for i in range(number_of_loc):
        distance += position_distance(loc_in_order[i], loc_in_order[i + 1])
    return distance
def distance_traveled():
    plot_info = {
        "title": "Distance traveled (km)",
        "yaxis": "Distance traveled (km)",
        "plot_name": "distance_traveled",
    }
    return plot_info, distance_traveled_calc

def new_locations_visited_calc(uid, day, bin_id, user_day_feature_dict):
    day_dict = user_day_feature_dict[uid]
    new_locations = plot_utils.unseen_features_for_prev_days(
        day_dict, "location", day, bin_id)
    return len(new_locations)
def new_locations_visited():
    plot_info = {
        "title": "New locations visited (Not visited in the last month)",
        "yaxis": "New locations visited",
        "plot_name": "number_of_new_location_visited",
    }
    return plot_info, new_locations_visited_calc

def get_top_location(day_dict, current_day, top, number_of_previous_days=30):
    """
    Top location is defined as the most frequent location visited based on days.
    How many days was the location visited in the past 30 days? 
    """
    prev_days = (current_day + dt.timedelta(x) 
        for x in range(-number_of_previous_days, 0)
        if current_day + dt.timedelta(x) in day_dict
    )
    locations = []
    for day in prev_days:
        for feature_dict in day_dict[day].values():
            locations.extend(feature_dict.get("location",{}).keys())
    top_loc = [loc for loc, _ in Counter(locations).most_common(top)]  
    return top_loc

def visits_top_n_location(uid, day, bin_id, user_day_feature_dict, top):
    day_dict = user_day_feature_dict[uid]
    top_locs = get_top_location(day_dict, day, top)
    if len(top_locs) < top:
        return np.nan
    if top_locs[top-1] in day_dict[day][bin_id].get("location", {}):
        return 1
    else:
        return 0

def visits_other_than_top_n_location(uid, day, bin_id, user_day_feature_dict, top):
    day_dict = user_day_feature_dict[uid]
    top_locs = set(get_top_location(day_dict, day, top))

    today_locations = day_dict[day][bin_id].get("location", {})
    if len(today_locations) > top:
        return 1
    if set(today_locations).issubset(top_locs):
        return 0
    else:
        return 1

def visits_non_top_1_location():
    plot_info = {
        "title": "Visited other than top 1 locations",
        "yaxis": "Visited non-top location",
        "plot_name": "visits_other_than_top_1_locations",
    }
    return plot_info, visits_other_than_top_n_location, 1
    
def visits_non_top_2_location():
    plot_info = {
        "title": "Visited other than top 2 locations",
        "yaxis": "Visited non-top location",
        "plot_name": "visits_other_than_top_2_locations",
    }
    return plot_info, visits_other_than_top_n_location, 2
def visits_non_top_3_location():
    plot_info = {
        "title": "Visited other than top 3 locations",
        "yaxis": "Visited non-top location",
        "plot_name": "visits_other_than_top_n_location",
    }
    return plot_info, visits_other_than_top_n_location, 3

def visits_top_1_location():
    plot_info = {
        "title": "Visited top 1 location",
        "yaxis": "Visited top location",
        "plot_name": "visits_top_1_location",
    }
    return plot_info, visits_top_n_location, 1
def visits_top_2_location():
    plot_info = {
        "title": "Visited top 2 location",
        "yaxis": "Visited top location",
        "plot_name": "visits_top_2_location",
    }
    return plot_info, visits_top_n_location, 2
def visits_top_3_location():
    plot_info = {
        "title": "Visited top 3 location",
        "yaxis": "Visited top location",
        "plot_name": "visits_top_3_location",
    }
    return plot_info, visits_top_n_location, 3



def get_last_visit_location(day_dict, current_day, location, number_of_previous_days=365*2):
    """
    How many days ago was the last visit to this location?
    """
    prev_days = ((current_day + dt.timedelta(-x), x)
        for x in range(1,number_of_previous_days)
        if current_day + dt.timedelta(-x) in day_dict
    )
    for day, num_days_ago in prev_days:
        for _, feature_dict in day_dict[day].items():
            if location in feature_dict.get("location", {}):
                return num_days_ago
    return number_of_previous_days

def time_since_last_visit_calc(uid, day, bin_id, user_day_feature_dict):
    day_dict = user_day_feature_dict[uid]
    today_locations = day_dict[day][bin_id].get("location", {})
    if len(today_locations) == 0:
        return np.nan

    vals = [
        get_last_visit_location(day_dict, day, loc) 
        for loc in today_locations
    ]
    max_val = max(vals)
    if max_val == -1:
        #print max_val, vals
        return np.nan
    return max_val

def time_since_last_visit():
    plot_info = {
        "title": "time_since_last_visit",
        "yaxis": "Visited top location",
        "plot_name": "time_since_last_visit",
    }
    return plot_info, time_since_last_visit_calc



# =====================================
#        Intermediate Time
# =====================================

"""
def plot_intermediate_call_and_text_time(uids, hours_per_bin):
def get_intermediate_time_list(uid, day, bin_id, features):
    all_records = []
    for feature in features:
        all_records += user_day_feature_dict[uid][day][bin_id][feature].values()
    # Flattening the list (taken from https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python)
    all_records = [
        dur_timestamp_tuple for sublist in all_records for dur_timestamp_tuple in sublist]
    # List of (start_time, end_time) for all phonecalls
    all_records_times = [(time, time + dt.timedelta(seconds=dur))
                            for dur, time in all_records]
    all_records_sorted = sorted(all_records_times, key=lambda x: x[0])
    inter_times = []
    for i in range(1, len(all_records_sorted)):
        inter_times.append(
            max(0, (all_records_sorted[i][0] - all_records_sorted[i - 1][1]).total_seconds()))
    return inter_times

def plot_intermediate_times(feature_list, feature_type, stat_type):
    if (all([x.endswith('in') for x in feature_list]) or
            all([x.endswith('out') for x in feature_list])):
        plot_type = " incoming" if feature_list[0].endswith(
            'in') else " outgoing"
    else:
        plot_type = ""
    plot_info = {
        "title": "%s of time between%s %ss" % (stat_type, plot_type, feature_type.replace("_", " ")),
        "yaxis": "Time between%s %ss" % (plot_type, feature_type.replace("_", " ")),
        "plot_name": "%s%s_%s" % (stat_type.lower(), "_" + plot_type.strip() if plot_type != "" else "", feature_type),
        "folder_name": "intermediate_times"
    }

    def calc_var(uid, day, bin_id):
        call_list = get_intermediate_time_list(
            uid, day, bin_id, feature_list)
        if stat_type == "Variance":
            return np.var(call_list) if len(call_list) > 0 else 0
        elif stat_type == "Mean":
            return np.mean(call_list) if len(call_list) > 0 else 0
    return plot_week(uids, plot_info, hours_per_bin, calc_var)

for stat_type in ["Variance", "Mean"]:
    plot_intermediate_times(["call_in"], "call", stat_type)
    plot_intermediate_times(["call_out"], "call", stat_type)
    plot_intermediate_times(["call_in", "call_out"], "call", stat_type)

    plot_intermediate_times(["text_in"], "text", stat_type)
    plot_intermediate_times(["text_out"], "text", stat_type)
    plot_intermediate_times(["text_in", "text_out"], "text", stat_type)

    plot_intermediate_times(["text_in", "call_in"],
                            "all_call_and_text", stat_type)
    plot_intermediate_times(
        ["text_out", "call_out"], "all_call_and_text", stat_type)
    plot_intermediate_times(
        ["text_in", "call_in", "text_out", "call_out"], "all_call_and_text", stat_type)
"""