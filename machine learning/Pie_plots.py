import os
import glob
import os.path as p
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

def fun(val):
    #print(val)
    return max(val)

for datatype, delay in [("absolute",1), ("absolute", 7), ("relative", 1)]:
    model_lookup = {
        0: 'Linear', 
        1: 'Dense',
        2: 'GRU', 
        3: 'Bidir-GRU',
    }

    feature_models = {
        1: [{} for _ in range(4)], 
        3: [{} for _ in range(4)]
    }
    files =  glob.glob('results%s/%s/**/*.tex' % 
        ("" if delay == 1 else "_7_day_delay", datatype), 
        recursive=True
    )
    for filename in sorted(files, key=len, reverse=True):
        features_bins_str = p.basename(p.dirname(p.dirname(filename))).split("_")
        features = int(features_bins_str[0])
        bins_str = ",".join([x.replace('h', '') for x in features_bins_str[2:]]) 
        print(features, bins_str, filename)
        with open(filename, "r") as f:
            f.readline()
            f.readline()
            f.readline()
            while True:
                line = f.readline().strip()
                if not line[0].isdigit():
                    break
                days = int(line[:line.index(" ")])
                try:
                    data = [float(x.strip()[:5]) for x in line.split("&")]
                    pre_data = data[2::2]
                    for model_id in range(len(pre_data)):
                        feature_models[features][model_id].setdefault((days, bins_str), []).append(pre_data[model_id])
                except Exception as ex:
                    print("Exception: ", ex)
                    continue

    print("Done Parseing tex files!")
    
    bins = ['24', '8,24,48', '4,8,24,48,72,96']
    max_size = max([
        max([fun(v) for (d,b), v in model.items() if b in bins])
        for models in feature_models.values()
        for model in models
    ])
    for features, models in feature_models.items():
        days_set, bin_set = set(), set()
        for m in models:
            for days, _ in m.keys():
                days_set.update({days})
        days = list(days_set)
        days.sort()
        print(days, bins)

        print(max_size)
        traces = []
        labels = []
        values = []

        for bin_i in range(len(bins)):   
            for day_i in range(len(days)):   
                key = (days[day_i], bins[bin_i])
                x_min, x_max = bin_i/len(bins), (bin_i+1)/len(bins)
                y_min, y_max = day_i/len(days), (day_i+1)/len(days)
                center_x, center_y = x_max - (x_max - x_min) / 2, y_max - (y_max - y_min) / 2

                try:
                    size = fun([ max(m[key]) for m in models])
                except:
                    continue
                delta_size = (size/max_size)
                traces.append({
                    "hole": 0.2, 
                    "direction": "clockwise", 
                    "sort": False, 
                    "labels": list(model_lookup.values()),
                    "type": "pie", 
                    "domain": {
                        "x": [
                            center_x - abs(x_min - center_x) * delta_size,
                            center_x + abs(x_max - center_x) * delta_size,
                        ], 
                        "y": [
                            center_y - (center_y - y_min) * delta_size,
                            center_y + (y_max - center_y) * delta_size,
                        ], 
                    },
                    "textfont": {"color":"#FFFFFF"},
                    "text": [
                        ("%0.2f" % fun(model[days[day_i],bins[bin_i]]))[1:]
                        for model in models
                    ],
                    "textinfo": "text",
                    "textposition": "inside", 
                    "values": [
                        fun(model[days[day_i],bins[bin_i]])
                        for model in models
                    ]
                })
        if datatype == "relative":
            annon_y = [-0.05433130699088572, -0.05347825675698572, -0.05460309752931429]
        else:
            annon_y = [-0.0811170212766, -0.0677639710427, -0.0778173832436]
        layout = go.Layout( 
            legend= {
                "x": 0, 
                "y": -0.103 if datatype == "relative" else -0.135, 
                "orientation": "h"
            }, 
            annotations=[
                {
                "x": -0.497115866882, 
                "y": -0.107712765957, 
                "showarrow": False, 
                "text": "<br>", 
                "xref": "x", 
                "yref": "paper"
                }, 
                {
                "x": -0.279920420102, 
                "y": -0.102393617021, 
                "showarrow": False, 
                "text": "<br>", 
                "xref": "x", 
                "yref": "paper"
                }, 
                {
                    "xref": "x",
                    "yref": "paper",
                    "text": "24h",
                    "y": annon_y[0],
                    "x": -0.499464409849713,
                    "showarrow": False
                },
                {
                    "xref": "x",
                    "yref": "paper",
                    "text": "8,24,48h",
                    "y": annon_y[2],
                    "x": -0.39867091917530006,
                    "showarrow": False
                },
                {
                    "xref": "x",
                    "yref": "paper",
                    "text": "4,8,24,48,72,96h",
                    "y": annon_y[1],
                    "x": -0.29720859889196516,
                    "showarrow": False
                }
            ],
            margin=dict(
                t=110,
                b=110,
                r=50 if features == 3 else 0,
                l=50 if features == 1 else 0
            ),
            width = 400,
            height = 600 if datatype == "absolute" else 750,
            title = "%s days<br>Labeled sick %s<br>PR-AUC for the %s" % (
                datatype.capitalize(), 
                "on the last day" if delay == 1 else "in last 7 days",
                "top feature"  if features == 1 else ("top %d features" % features)),
            xaxis = {
                "autorange": False, 
                "gridwidth": 1, 
                "mirror": False, 
                "range": [-0.551129325614226, -0.24711586701335952],  
                "showgrid": False, 
                "showline": True, 
                "showticklabels": False, 
                "side": "bottom", 
                "tickangle": 90, 
                "title": "<br>Bin sizes",  
                "zeroline": False
            }, 
            showlegend=features == 1, 
            yaxis={
                "autorange": False, 
                "showgrid": False, 
                "showline": True, 
                "side": "right" if features == 3 else "left", 
                "showticklabels": features == 1, 
                "range": [3.00858707231, 32.8278729915] if datatype == "absolute" else [4.480517322828405, 12.496851139591888], 
                "title": "Subsequence length (days)" if features == 1 else ""
            }
        )

        py.sign_in('thorgeirk11', '??')  # Pro features
        fig = go.Figure(data=traces, layout=layout)
        py.image.save_as(fig, 'pie_chart%s_%s_%d_features.pdf' % ("" if delay == 1 else "_7DayDelay", datatype,features))
        #print(py.plot(fig, filename='pie_chart2_%s_%d_features' % (datatype,features)))
