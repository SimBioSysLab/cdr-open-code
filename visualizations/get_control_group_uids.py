import pandas as pd
import numpy as np
import ciso8601
import os
import random as rand
import datetime as dt
import multiprocessing as mp
from utils.CDR_csv_reader_loc_only import DataReader


# Fiter out uids if they have too few records for the onset week.
NUMBER_OF_ACTIVE_DAYS_IN_ONSET_WEEK = 3

# File selection macros
HOME_TOWER_FILE = 'home_towers.csv'
ONSET_DATA_FILE = '../../inputData/FULLonset.csv'
G1_DATA_FILE = '../../inputData/g1calls.csv'             # All records from G1
LARGE_DATA_FILE = '../../inputData/FULLg1sickcalls.csv'  # Only sick individuals from G1
SMALL_DATA_FILE = '../../inputData/SMALLg1sickcalls.csv' # Subset of sick individuals from G1

SICK_CALL_DATA_FILE, FILE_IN_USE = (LARGE_DATA_FILE, "large")
#SICK_CALL_DATA_FILE, FILE_IN_USE = (SMALL_DATA_FILE, "small")

# List of bin sizes to run
HOURS_PER_BIN_LIST = [8]

# Four weeks
ONSET_PERIOD = range(-14,15)

NUMBER_OF_CONTROL_GROUPS = 3

# =====================================
#           Read in the data
# =====================================

def read_home_tower_csv():
    FULLonset = '../../inputData/FULLonset.csv'
    exclude_uids = pd.read_csv(FULLonset, usecols=[0]).values

    data = pd.read_csv(HOME_TOWER_FILE, na_values='', usecols=[1, 4])
    home_towers = {}
    for uid, tower_id in data.values:
        if uid in exclude_uids:
            continue
        if tower_id is None:
            continue
        if tower_id not in home_towers:
            home_towers[tower_id] = []
        home_towers[tower_id].append(uid)  
    return home_towers

def read_data():
    fitered_uids_path = 'sick_uids_to_use_%s.txt' % FILE_IN_USE
    if os.path.isfile(fitered_uids_path):
        sick_uids = set(np.loadtxt(fitered_uids_path))
    else:
        print "PLEASE GENIRATE"
        return

    data_reader = DataReader(SICK_CALL_DATA_FILE, ONSET_DATA_FILE)

    print "Reading %s data file" % FILE_IN_USE
    user_dicts = data_reader.read_call_data(sick_uids)
    onset_dates_dict = data_reader.read_onset_data()

    print "Number of uids in user_dicts", len(user_dicts)
    print "Number of uids in onset_dates_dict", sum([len(uids) for uids in onset_dates_dict.values()])
    return onset_dates_dict, user_dicts
# =====================================
#           Find similar uids
# =====================================

def home_tower(user_dict, onset_date):
    onset_days = [onset_date + dt.timedelta(x) for x in ONSET_PERIOD]
    location_dict = {}
    for day in onset_days:
        if day in user_dict:
            for loc, count in user_dict[day].items():
                if loc not in location_dict:
                    location_dict[loc] = 0
                location_dict[loc] += count

    if len(location_dict) == 0:
        return None
    
    v = list(location_dict.values())
    k = list(location_dict.keys()) 
    return k[v.index(max(v))]
    
def main():
    onset_dates_dict, sick_users_dict = read_data()

    sick_multiple_control_groups_mappings = {}
    np.random.seed(654333)
    num = 0
    for _ in range(NUMBER_OF_CONTROL_GROUPS):
        healthy_uids_with_home_towers = read_home_tower_csv()
        # Each iteration contains single onset date and all uids 
        for onset_date, sick_uids in onset_dates_dict.items():
            onset_str = onset_date.strftime("%A_%d_%B_%Y")
            
            sick_home_towers = {}

            # Iterate through sick uids on this onset date
            # and find their home towers
            for uid in sick_uids:
                # Skip (sick) users that don't have enough data
                if uid not in sick_users_dict:
                    continue
                home_tower_id = home_tower(sick_users_dict[uid], onset_date)
                if home_tower_id is None:
                    continue
                if home_tower_id not in sick_home_towers:
                    sick_home_towers[home_tower_id] = []
                sick_home_towers[home_tower_id].append(uid)
            
            for tower_id, sick_uids_for_tower in sick_home_towers.items():
                if tower_id in healthy_uids_with_home_towers:
                    healthy_users_with_same_home_tower = healthy_uids_with_home_towers[tower_id]
                    number_of_sick = len(sick_uids_for_tower)
                    number_of_healthy =  len(healthy_users_with_same_home_tower)
                    # Make sure we have enough healthy people with the same home tower as the sick people
                    assert number_of_healthy >= number_of_sick
                    healthy_sample = np.random.choice(healthy_uids_with_home_towers[tower_id], size=number_of_sick, replace=False)

                    for i in range(number_of_sick):
                        # Find healthy user id and remove it from healthy users dict
                        healthy_uid = healthy_sample[i]
                        index_of_healthy = healthy_uids_with_home_towers[tower_id].index(healthy_uid)
                        del healthy_uids_with_home_towers[tower_id][index_of_healthy]
                        sick_uid = sick_uids_for_tower[i]
                        if sick_uid not in sick_multiple_control_groups_mappings:
                            sick_multiple_control_groups_mappings[sick_uid] = []
                        else:
                            if len(sick_multiple_control_groups_mappings[sick_uid]) == NUMBER_OF_CONTROL_GROUPS:
                                continue
                            prev_healthy_uids = [val[1] for val in sick_multiple_control_groups_mappings[sick_uid]]
                            found_healthy_uid_suitable = False
                            for j in range(len(healthy_sample)):
                                if healthy_uid in prev_healthy_uids:
                                    healthy_uid = healthy_sample[j]
                                else:
                                    found_healthy_uid_suitable = True
                                    break
                            if not found_healthy_uid_suitable:
                                #print "NO SUITABLE HEALTHY ID FOUND!"
                                num += 1
                        date_and_matched_healthy_uid = (onset_date, healthy_uid)
                        sick_multiple_control_groups_mappings[sick_uid].append(date_and_matched_healthy_uid)

    print num, "number of problematic mappings"
    # Sanity checking code
    sick_uids = [sick_uid for sick_uid in sick_multiple_control_groups_mappings.keys()]
    assert len(sick_uids) == len(set(sick_uids))
    assert all(len(val) == NUMBER_OF_CONTROL_GROUPS for val in sick_multiple_control_groups_mappings.values())
    for i in range(NUMBER_OF_CONTROL_GROUPS):
        healthy_uids = [val[i] for val in sick_multiple_control_groups_mappings.values()]
        #assert len(healthy_uids) == len(set(healthy_uids))
        intersect = set(sick_uids) & set(healthy_uids)
        assert intersect == set([])

    l = len(set(sick_uids + healthy_uids))
    print len(sick_uids), len(healthy_uids), l, len(set(sick_uids)), len(set(healthy_uids))
    if l < len(sick_uids) + len(healthy_uids):
        print "DUPLICATE found!!"

    mapping_2d = []
    for sick_uid, arr in sick_multiple_control_groups_mappings.items():
        control_uids = [val[1] for val in arr]
        onset_date = arr[0][0]
        mapping = [sick_uid, onset_date]
        mapping.extend(control_uids)
        mapping_2d.append(mapping)
    np.savetxt("healthy_sick_pair_uid_mapping.csv", mapping_2d, fmt='%s', delimiter=',')

if __name__ == "__main__":
    main()
