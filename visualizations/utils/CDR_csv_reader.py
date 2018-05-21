import pandas as pd
import numpy as np
import datetime as dt
import ciso8601
import math
import mmap
import gc
from tqdm import tqdm

CHUNK_SIZE = 10**5
#CHUNK_SIZE = None

class DataReader:
    def __init__(self, hours_per_bin_list, onset_and_control_group_mapping_file=None, call_data_file=None, onset_data_file=None, extra_location_file=None):
        self.data_file = call_data_file
        self.hours_per_bin_list = hours_per_bin_list
        self.onset_data_file = onset_data_file
        self.extra_location_file = extra_location_file
        self.control_group_mapping_file = onset_and_control_group_mapping_file

    # =====================================
    #             Utilities
    # =====================================
    def get_num_lines(self, file):
        with open(file) as f:
            for i, _ in enumerate(f):
                pass
            return i + 1
    # =====================================
    #             Read Data
    # =====================================
    

    def parse(self, hours_per_bin_dict, file_path, uids_to_use, use_date_period, collect_loc, collect_call_text):
        should_exclude_uid = uids_to_use is not None and len(uids_to_use) > 0 
        should_exclude_date = use_date_period is not None and len(use_date_period) > 0 
        
        feature_name = {
            ('t', 't'): 'call_in',
            ('f', 't'): 'call_out',
            ('t', 'f'): 'text_in',
            ('f', 'f'): 'text_out',
        }
        print("Reading: ", file_path)
        with open(file_path) as csv_file:
            lines = enumerate(csv_file)
            next(lines) # Read header
            for _, line in tqdm(lines, desc="Reading", unit="lines", total=self.get_num_lines(file_path)):
                vals = line.rstrip().rsplit(',')
                if len(vals) == 11:
                    uid, obj, timestamp, is_in, is_call, _, _, units, _, lat, lon = vals
                elif len(vals) > 0:
                    uid, timestamp, _, _, _, lat, lon = vals
                else:
                    break
                uid = int(uid)
                if should_exclude_uid and uid not in uids_to_use:
                    continue
                timestamp = ciso8601.parse_datetime(timestamp)
                date = timestamp.date()
                # Iterate through different bin sizes
                for hours_per_bin in self.hours_per_bin_list:
                    for day_delta in range(max(1, hours_per_bin // 24)):
                        
                        day = date + dt.timedelta(day_delta)
                        if should_exclude_date and day not in use_date_period:
                            break
                        # Initialize dictionary for current hour-bin size
                        uid_dict = hours_per_bin_dict.setdefault(hours_per_bin, {})
                        day_dict = uid_dict.setdefault(uid, {})
                        bin_dict = day_dict.setdefault(day, {})
                        bin_id = timestamp.hour // hours_per_bin
                        feature_dict = bin_dict.setdefault(bin_id, {})

                        # Add location, call, and text data to the dictionary
                        if collect_loc and lat != '':
                            coord = float(lat), float(lon)
                            loc_dict = feature_dict.setdefault('location', {})
                            loc_dict.setdefault(coord, []).append(timestamp)
                        if collect_call_text:
                            obj, units = int(obj), int(units)  
                            call_text_dict = feature_dict.setdefault(feature_name[(is_in, is_call)], {})
                            call_text_dict.setdefault(obj, []).append((units, timestamp))
        return hours_per_bin_dict

    def read_extra_loc_data(self, hours_per_bin_dict, uids_to_use=None, use_date_period=None):
        return self.parse(hours_per_bin_dict, self.extra_location_file, 
            uids_to_use, use_date_period, True, False)

    def read_call_data(self, hours_per_bin_dict, file_path, uids_to_use=None, use_date_period=None, collect_loc=True, collect_call_text=True):
        return self.parse(hours_per_bin_dict, file_path if file_path is not None else self.data_file,
            uids_to_use, use_date_period, collect_loc, collect_call_text)

    def read_onset_data(self, uids_to_use):
        # Initialize onset data dictionary
        uids_not_empty = uids_to_use is not None and len(uids_to_use) > 0
        onset_dict = {}
        data_frame = pd.read_csv(self.onset_data_file)
        for uid, day in data_frame.values:
            if uids_not_empty and uid not in uids_to_use:
                continue
            # Parse day from timestamp into datetime object
            day = ciso8601.parse_datetime(day).date()
            # Initialize user id list for current diagnosis day
            if day not in onset_dict:
                onset_dict[day] = []
            # Add user to onset dictionary
            onset_dict[day].append(uid)
        return list(onset_dict.items())

    def read_onset_and_control_group_mapping_file(self, uids_to_use=None, control_group_count=3):
        uids_not_empty = uids_to_use is not None and len(uids_to_use) > 0
        data_frame = pd.read_csv(self.control_group_mapping_file)
        onset_dicts = [] 
        for index in range(control_group_count+1):
            onset_dict = {}
            for vals in data_frame.values:
                sick_uid = vals[0]
                if uids_not_empty and sick_uid not in uids_to_use:
                    continue
                day = vals[1]
                uids = list([sick_uid]) + list(vals[2:])
                # Parse day from timestamp into datetime object
                day = ciso8601.parse_datetime(day).date()
                # Initialize user id list for current diagnosis day
                if day not in onset_dict:
                    onset_dict[day] = []
                # Add user to onset dictionary
                onset_dict[day].append(uids[index])
            onset_dicts.append(onset_dict)
        return onset_dicts

    def remove_uids_with_no_data_during_onset_week(self, onset_period, day_uids_onset, user_day_feature_dict, number_of_active_days_in_onset_week):
        onset_and_uids_withdata = {}
        # Loop through onset data dictionary
        for onset_date, uids in day_uids_onset.items():
            new_uids = []
            # Construct list of datetime objects that represent DoD +/- 3 days
            onset_week = [onset_date + dt.timedelta(x) for x in onset_period]
            for uid in uids:
                # Count days that contain any data for current user
                days_with_data = len(
                    [day for day in onset_week if day in user_day_feature_dict[uid]])
                if days_with_data >= number_of_active_days_in_onset_week:
                    new_uids.append(uid)
            if len(new_uids) > 0:
                onset_and_uids_withdata[onset_date] = new_uids
        return onset_and_uids_withdata