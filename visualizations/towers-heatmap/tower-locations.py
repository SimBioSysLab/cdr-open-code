import gmplot
import pandas as pd
import numpy as np
import os
import random
import ciso8601
import datetime as dt
import mmap
from tqdm import tqdm

COORD_FILE = 'tower_coords.txt'
# File selection macros
G1_DATA_FILE = '../../../inputData/large/g1calls_diagnosed.csv'    # All records from G1
LARGE_DATA_FILE = '../../../inputData/FULLg1sickcalls.csv'  # Only sick individuals from G1
SMALL_DATA_FILE = '../../../inputData/SMALLg1sickcalls.csv' # Subset of sick individuals from G1

CALL_DATA_FILE, FILE_IN_USE = (G1_DATA_FILE, "large")
#CALL_DATA_FILE, FILE_IN_USE = (LARGE_DATA_FILE, "large")
#CALL_DATA_FILE, FILE_IN_USE = (SMALL_DATA_FILE, "small")

CHUNK_SIZE = 10**6
FUZZ_VAL = 0.02

def get_num_lines():
    with open(CALL_DATA_FILE, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        lines = 0
        while mm.readline():
            lines += 1
        del mm
        return lines

def read_person_movement_data(uid_to_use, start_date, end_date):
    coordinates = {}
    num_days = (end_date - start_date).days
    # If user coordinates haven't been save to file, we need to fetch them from 
    # the .csv data file and write them to a file
    with open(CALL_DATA_FILE, 'r+b') as csv_file:
        mm = mmap.mmap(csv_file.fileno(), 0, prot=mmap.PROT_READ)
        header = mm.readline()
        for line in tqdm(iter(mm.readline, ""), desc="Reading", unit="lines", total=get_num_lines()):
            vals = line.strip("\n").split(',')
            # header: subject, object,  timestamp,           in, call, tarif, tariftype, units, towerid, lat,       lon
            # values: 5576595, 4573172, 2009-02-01 00:00:01, t,  f,    ,      PREP,      0,     3501,    65.679166, -18.092559
            # values: 9536617, 4373959, 2009-02-01 00:00:02, f,  t,    GGSM7, POST,      1,     3210,    64.12892,  -21.8287

            # Extract relevant values from record in file
            uid, obj, timestamp, is_in, is_call, _, _, units, _, lat, lon = vals
            uid = int(uid)
            if uid_to_use != uid:
                continue
            if lat == "":
                continue
            lat, lon = float(lat), float(lon)
            if np.isnan(lat) or np.isnan(lon):
                continue

            timestamp = ciso8601.parse_datetime(timestamp)
            if not (start_date < timestamp < end_date):
                continue

            if (lat, lon) not in coordinates:
                coordinates[(lat, lon)] = []
            coordinates[(lat, lon)].append(timestamp)

    return coordinates

def read_person_location_data(uid_to_use):
    data_frame = pd.read_csv(CALL_DATA_FILE, na_values='', chunksize=CHUNK_SIZE)
    i = 0
    coordinates = {}
    filename = '%d_coordinates.txt' % uid_to_use
    # If coordinates for user already exist just read the file
    if os.path.isfile(filename):
        with open(filename, 'r') as user_data_file:
            for line in user_data_file.read().split('\n'):
            	if line != "":
                	count, lat, lon = line.split(',')
                	coordinates[(float(lat), float(lon))] = int(count)
        return coordinates

    # If user coordinates haven't been save to file, we need to fetch them from 
    # the .csv data file and write them to a file
    for chunk in data_frame:
        for vals in chunk.values:
            # header: subject, object,  timestamp,           in, call, tarif, tariftype, units, towerid, lat,       lon
            # values: 5576595, 4573172, 2009-02-01 00:00:01, t,  f,    ,      PREP,      0,     3501,    65.679166, -18.092559
            # values: 9536617, 4373959, 2009-02-01 00:00:02, f,  t,    GGSM7, POST,      1,     3210,    64.12892,  -21.8287

            # Extract relevant values from record in file
            uid, obj, timestamp, is_in, is_call, _, _, units, _, lat, lon = vals
            if uid_to_use != uid:
                continue
            if np.isnan(lat) or np.isnan(lon):
                continue

            #print 'append uid %d wit coordinates %f, %f' % (uid, lat, lon)
            if (lat, lon) not in coordinates:
                coordinates[(lat, lon)] = 0
            coordinates[(lat, lon)] += 1

        i += 1
        print "Chunk %d done" % i
    with open(filename, 'w') as user_data_file:
    	for (lat, lon), count in coordinates.items():
    		user_data_file.write('%d,%f,%f\n' % (count, lat, lon))

    return coordinates

def read_tower_coords():
    with open(COORD_FILE, 'r') as coord_file:
        coords = coord_file.read().split('\n')
        lats, lons = [], []
        for c in coords:
            try:
                lat, lon = c.split(',')
                lat, lon = float(lat), float(lon)
                lats.append(lat)
                lons.append(lon)
            except ValueError as err:
                continue
        return lats, lons

def fuzz_coord_value(val):
    fuzziness = random.uniform(-FUZZ_VAL, FUZZ_VAL)
    return val + fuzziness

def fuzz_coordinates(coordinates=None):
    if coordinates is None:
        coordinates = zip(*read_tower_coords())
    fuzzed_lats = []
    fuzzed_lons = []
    for lat, lon in coordinates:
        fuzzed_lat = fuzz_coord_value(lat)
        fuzzed_lon = fuzz_coord_value(lon)
        fuzzed_lats.append(fuzzed_lat)
        fuzzed_lons.append(fuzzed_lon)
    return fuzzed_lats, fuzzed_lons

def remove_adjacent_coordinates(coordinates):
    DISTANCE = 0.01
    delete = False
    lats, lons = [], []
    for lat, lon in coordinates:
        for lat_dup, lon_dup in coordinates:
            if lat == lat_dup and lon == lon_dup:
                continue
            lat_diff = abs(lat - lat_dup)
            lon_diff = abs(lon - lon_dup)
            if lat_diff < DISTANCE and lon_diff < DISTANCE:
                delete = True
                break
        if delete:
            delete = False
            continue
        lats.append(lat)
        lons.append(lon)
    print "Removed %d coordinates" % (len(coordinates) - len(lats))
    return lats, lons

def generate_movement_and_scatter(user_coord_dict, sc_lats, sc_lons, filename):
    #thresholds = [0, 1, 10, 50, 200, 1000]
    #radii = [10, 15, 20, 100]
    radius = 15
    zoom = 7
    gmap = gmplot.GoogleMapPlotter(65, -19, zoom)
    
    size = 1500
    gmap.scatter(sc_lats, sc_lons, 'k', marker=True)

    threshold = 100
    #radius = 10
    opacity = 0.8
    base_colors = [
        (255,0,0,0),
        (0,255,0,0),
        (0,0,255,0)
    ]
    extra_colors = [
        (255,0,0,1),
        (0,255,0,1),
        (0,0,255,1)
    ]
    timestamp_first = dt.datetime(2010, 3, 18, 4, 0, 0)
    timestamp_second = dt.datetime(2010, 3, 19, 17, 0, 0)
    timestamp_third = dt.datetime(2010, 3, 19, 23, 59, 59)
    timestamp_fourth = dt.datetime(2010, 3, 20, 23, 59, 59)
    coord_days = [[], [], []]
    for coords, timestamp_list in user_coord_dict.items():
        for timestamp in timestamp_list:
            # First color in heatmap
            if timestamp_first < timestamp < timestamp_second:
                coord_days[0].append(coords)
            # Second color in heatmap
            elif timestamp_second < timestamp < timestamp_third:
                coord_days[1].append(coords)
            # Third color in heatmap
            elif timestamp_third < timestamp < timestamp_fourth:
                coord_days[2].append(coords)
            # Edge case that shouldn't happen
            else:
                raise Exception("!Found case that should NOT occur!")
    gradient_magic_value = max([len(arr) for arr in user_coord_dict.values()])
    for i in range(len(coord_days)):
        coords = coord_days[i]
        gradient = []
        gradient.append(base_colors[i])
        for _ in range(int(0.9 * gradient_magic_value)):
            gradient.append(extra_colors[i])
        lats, lons = zip(*coords)
        gmap.heatmap(lats, lons,
            threshold=threshold, radius=radius, gradient=gradient, opacity=opacity)

    gmap.draw('%s_%d.html' % (filename, radius))

def generate_heatmap_and_scatter(user_coord_dict, sc_lats, sc_lons, filename):
    lats, lons = zip(*user_coord_dict.keys())

    #thresholds = [0, 1, 10, 50, 200, 1000]
    radii = [10, 15, 20, 100]
    for radius in radii:
        zoom = 7
        gmap = gmplot.GoogleMapPlotter(65, -19, zoom)
        
        size = 1500
        gmap.scatter(sc_lats, sc_lons, 'k', marker=True)

        threshold = 100
        #radius = 10
        opacity = 0.8
        gradient = [
            (0,255,0,0)
        ]
        gradient_magic_value = max(user_coord_dict.values())/10
        # Gradient colors #00FF00 -> #48FF00 -> #91FF00 -> #DAFF00 -> #FF9100
        #                    -> #FF4800 -> #FF0000
        for _ in range(int(0.03 * gradient_magic_value)):
            gradient.append((0,255,0,0.9))
        for _ in range(int(0.04 * gradient_magic_value)):
            gradient.append((72,255,0,1))
        for _ in range(int(0.04 * gradient_magic_value)):
            gradient.append((145,255,0,1))
        for _ in range(int(0.04 * gradient_magic_value)):
            gradient.append((218,255,0,1))
        for _ in range(int(0.04 * gradient_magic_value)):
            gradient.append((255,218,0,1))
        for _ in range(int(0.04 * gradient_magic_value)):
            gradient.append((255,145,0,1))
        for _ in range(int(0.08     * gradient_magic_value)):
            gradient.append((255,72,0,1))
        for _ in range(int(0.69 * gradient_magic_value)):
            gradient.append((255,0,0,1))
        gmap.heatmap(lats, lons,
            threshold=threshold, radius=radius, gradient=gradient, opacity=opacity)

        #gmap.draw('%s_radius_%d.html' % (filename, radius))
        #gmap.draw('%s_threshold_%d.html' % (filename, threshold))
        #gmap.draw('%s_dissipating.html' % (filename))
        gmap.draw('%s_%d.html' % (filename, radius))

def generate_heatmap(lats, lons, filename):
    zoom = 7
    gmap = gmplot.GoogleMapPlotter(65, -19, zoom)

    gmap.heatmap(lats, lons, threshold=10, radius=12, gradient=None, opacity=0.8)

    gmap.draw('%s.html' % filename)

def generate_scatter(lats, lons, filename):
    zoom = 7
    gmap = gmplot.GoogleMapPlotter(65, -19, zoom)
    
    #gmap.scatter(lats, lons, 'k', marker=True)
    gmap.scatter(lats, lons, color='#k', size=1000, marker=False)

    gmap.draw('%s.html' % filename)

def single_person_heatmap():
    #uids = [8104827, 8244068, 4280187]
    #uids = [4280187]
    #uids = [8244068]
    uids = [8104827]
    start_date = dt.datetime(2010, 3, 18, 4, 0, 0)
    end_date = dt.datetime(2010, 3, 20, 23, 59, 59)
    tower_lats, tower_lons = read_tower_coords()
    for uid in uids:
        #user_coord_dict = read_person_location_data(uid)
        user_coord_dict = read_person_movement_data(uid, start_date, end_date)
        user_lats, user_lons = zip(*user_coord_dict.keys())
        unused_lats = [lat for lat in tower_lats if lat not in user_lats]
        unused_lons = [lon for lon in tower_lons if lon not in user_lons]
        unused_lats, unused_lons = remove_adjacent_coordinates(zip(unused_lats, unused_lons))
        fuzzed_lats, fuzzed_lons = fuzz_coordinates(zip(unused_lats, unused_lons))
        fuzz_coordinates(user_coord_dict.keys())
        filename = 'user_%d_movement' % uid
        #generate_heatmap(user_lats, user_lons, filename)
        #generate_scatter(tower_lats, tower_lons, 'tower_locations')
        generate_movement_and_scatter(user_coord_dict, fuzzed_lats, fuzzed_lons, filename)
        #generate_heatmap_and_scatter(user_coord_dict, unused_lats, unused_lons, filename)

def tower_distribution_heatmap():
    tower_lats, tower_lons = read_tower_coords()
    generate_heatmap(tower_lats, tower_lons, 'towers')

def main():
    single_person_heatmap()
    #lats, lons = fuzz_coordinates()
    #generate_scatter(lats, lons, 'towers-no-markers')

if __name__ == '__main__':
    main()