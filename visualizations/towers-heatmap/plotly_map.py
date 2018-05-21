import plotly.plotly as py
import plotly.graph_objs as go
import os
import random
import ciso8601
import datetime as dt
import mmap
import numpy as np
import exceptions
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
FUZZ_VAL = 0.1

def get_num_lines():
    with open(CALL_DATA_FILE, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        lines = 0
        while mm.readline():
            lines += 1
        del mm
        return lines

def read_person_movement_data_list(uid_to_use, start_date, end_date):
    coordinates = []
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

            if not (lat, lon) in [coord for _, coord in coordinates]:
                coordinates.append((timestamp, (lat, lon)))
            #if (lat, lon) not in coordinates:
            #    coordinates.append((lat, lon))
            #if (lat, lon) not in coordinates:
            #    coordinates[(lat, lon)] = []
            #coordinates[(lat, lon)].append(timestamp)

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

    # If user coordinates haven't been saved to file, we need to fetch them from 
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

def fuzz_user_coordinates(user_coord_list):
    fuzzed_user_coord_list = []
    for timestamp, coord in user_coord_list:
        lat, lon = coord
        fuzzed_lat = fuzz_coord_value(lat)
        fuzzed_lon = fuzz_coord_value(lon)
        fuzzed_user_coord_list.append((timestamp, (fuzzed_lat, fuzzed_lon)))
    return fuzzed_user_coord_list

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

def get_layout():
    return dict(
        font = dict(
            color = 'rgb(0,0,0)'
        ),
        title = '<b>Cellphone towers (and movement inference) in Iceland</b>',
        titlefont = dict(
            size = 29
        ),
        autosize = False,
        width = 800,
        height = 580,
        margin = go.Margin(
            l = 0,
            r = 0,
            t = 50,
            b = 0,
            pad = 0
        ),
        legend = dict(
            traceorder = 'grouped',
            orientation = 'v',
            borderwidth = 1,
            font = dict(
                size = 21
            ),
            x = 0.94,
            xanchor = 'center',
            y = 0.09,
            yanchor = 'middle'
        ),
        geo = dict(
            resolution = '50',
            projection = dict(
                type = 'mercator'
                ),
            lonaxis = dict(
                range = [-24.738867, -13.291112]
            ),
            lataxis = dict(
                range = [63.285559, 66.650824]
            ),
            showland = True,
            landcolor = 'white',
            showocean = True,
            oceancolor = 'rgb(216, 216, 216)',
            showlakes = True,
            lakecolor = 'rgb(216, 216, 216)',
            showrivers = True,
            rivercolor = 'rgb(216, 216, 216)'
        )
    )

def get_tower_map(tower_lats, tower_lons):
    towers = [ dict(
        type = 'scattergeo',
        lon = tower_lons,
        lat = tower_lats,
        mode = 'markers',
        marker = dict( 
            symbol = 'x',
            size = 8,
            color = 'black',
        ),
        legendgroup = 'towers_group',
        name = 'Cellphone towers',
        showlegend = True
    )]

    return towers

def determine_group_attributes(timestamp):
    attributes = {}
    timestamp_first = dt.datetime(2010, 3, 18, 4, 0, 0)
    timestamp_second = dt.datetime(2010, 3, 19, 17, 0, 0)
    timestamp_third = dt.datetime(2010, 3, 19, 23, 59, 59)
    timestamp_fourth = dt.datetime(2010, 3, 20, 23, 59, 59)
    group = 0
    # First time period in heatmap
    if timestamp_first < timestamp < timestamp_second:
        attributes['group_index'] = 0
        attributes['symbol'] = 'circle'
        attributes['color'] = 'rgb(230,97,1)'
        attributes['line_style'] = 'solid'
        attributes['legend_group'] = 'movement_group_1'
        attributes['legend_name'] = 'Movement Period 1'
    # Second time period in heatmap
    elif timestamp_second < timestamp < timestamp_third:
        attributes['group_index'] = 1
        attributes['symbol'] = 'triangle-up'
        attributes['color'] = 'rgb(153,142,195)'
        attributes['line_style'] = 'dash'
        attributes['legend_group'] = 'movement_group_2'
        attributes['legend_name'] = 'Movement Period 2'
    # Third time period in heatmap
    elif timestamp_third < timestamp < timestamp_fourth:
        attributes['group_index'] = 2
        attributes['symbol'] = 'square'
        attributes['color'] = 'rgb(230,97,1)'
        attributes['line_style'] = 'dot'
        attributes['legend_group'] = 'movement_group_3'
        attributes['legend_name'] = 'Movement Period 3'
    # Edge case that shouldn't happen
    else:
        raise Exception("!Found case that should NOT occur!")
    return attributes

def get_user_heatmap(user_coord_list):
    user_movement = []
    group_seen = [False, False, False]
    for i in range( len( user_coord_list ) - 1 ):
        timestamp = user_coord_list[i][0]
        group_attributes = determine_group_attributes(timestamp)
        if not group_seen[group_attributes['group_index']]:
            group_seen[group_attributes['group_index']] = True
            show_legend = True
        else:
            show_legend = False
        user_coord_start = user_coord_list[i][1]
        user_coord_end = user_coord_list[i+1][1]
        user_movement.append(
            dict(
                type = 'scattergeo',
                lon = [ user_coord_start[1], user_coord_end[1] ],
                lat = [ user_coord_start[0], user_coord_end[0] ],
                mode = 'lines+markers',
                line = dict(
                    width = 4,
                    dash = group_attributes['line_style'],
                    color = group_attributes['color'],
                ),
                marker = dict(
                    symbol = group_attributes['symbol'],
                    size = 8, 
                    color = group_attributes['color'],
                    line = dict(
                        width = 1,
                        color ='rgba(0, 0, 0, 0)'
                    )
                ),
                legendgroup = group_attributes['legend_group'],
                name = group_attributes['legend_name'],
                showlegend = show_legend
            )
        )

    return user_movement

def generate_map(iteration):
    uid = 8104827
    start_date = dt.datetime(2010, 3, 18, 4, 0, 0)
    end_date = dt.datetime(2010, 3, 20, 23, 59, 59)
    print("Reading tower coordinates")
    tower_lats, tower_lons = read_tower_coords()

    print("Reading user movement data")
    user_coord_list = read_person_movement_data_list(uid, start_date, end_date)
    user_lats, user_lons = zip(*[coord for _, coord in user_coord_list])
    print("Removing towers that user has visited")
    unused_tower_lats = [lat for lat in tower_lats if lat not in user_lats]
    unused_tower_lons = [lon for lon in tower_lons if lon not in user_lons]
    print("Removing duplicate/adjacent tower coordinates")
    filtered_tower_lats, filtered_tower_lons = remove_adjacent_coordinates(zip(unused_tower_lats, unused_tower_lons))
    print("Fuzzing tower coordinates")
    fuzzed_tower_lats, fuzzed_tower_lons = fuzz_coordinates(zip(filtered_tower_lats, filtered_tower_lons))
    print("Fuzzing user coordinates")
    fuzzed_user_coord_list = fuzz_user_coordinates(user_coord_list)

    tower_map = get_tower_map(fuzzed_tower_lats, fuzzed_tower_lons)
    user_movement_map = get_user_heatmap(fuzzed_user_coord_list)
    layout = get_layout()

    fig = dict(
        data=tower_map + user_movement_map,
        layout=layout
    )
    #filename = "maps/plotly_map.pdf"
    filename = "maps/plotly_map_it_%d.pdf" % iteration
    #filename = "plotly_map_it_%d.png" % iteration
    py.plot(fig, filename=filename)
    print("saving plot to %s" % filename)
    try:
        py.image.save_as(fig, filename=filename)
    except Exception as ex:
        print ex.message
        print ex.status_code
        print ex.content

def main():
    print("signing in to plotly")
    # Replace the username, and API key with your credentials.
    py.sign_in('thorgeirk11', '??')  # Pro features
    #py.sign_in('asfalt', '??')
    #py.sign_in('fravarpa', '??')

    for it in range(5):
        generate_map(it)
    #generate_map()

if __name__ == "__main__":
    main()