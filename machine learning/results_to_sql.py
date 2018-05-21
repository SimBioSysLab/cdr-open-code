import os
import glob
import os.path as p
import numpy as ny
import plotly as pl



with open("db_inserts.sql", "w+") as sql_file:
    sql_file.writelines("""
INSERT INTO Results (datatype, feature_count, bins, num_days, model, auc, per, test_n)
VALUES """)


    for filename in glob.glob('results/**/*.tex', recursive=True):
        #if p.isfile(filename):
        #    new_name = p.basename(p.dirname(filename))
        #    old_name = p.basename(filename)
        #    print(filename, old_name, new_name)
        #    os.rename(filename, filename.replace(old_name, new_name) + ".tex")
        test_n = int(p.basename(p.dirname(filename))[-1])
        relative_or_absolute = p.basename(p.dirname(p.dirname(p.dirname(filename))))
        features_bins_str = p.basename(p.dirname(p.dirname(filename))).split("_")
        features = int(features_bins_str[0])
        bins = " ".join(features_bins_str[2:])
        print(filename, features, bins)
        with open(filename, "r") as f:
            f.readline()
            f.readline()
            f.readline()
            for days in range(5,13): 
                try:
                    data = [float(x.strip().replace("\\\\","")) for x in f.readline().split("&")]
                    auc_data = data[1::2]
                    pre_data = data[2::2]
                    print(data)
                    
                    for i in range(len(auc_data)):
                        sql_file.write("('%s', %d, '%s', %d, %s, %0.3f, %0.3f, %d),\n" % (
                            relative_or_absolute,
                            features,
                            bins,
                            days,
                            i+1,
                            auc_data[i],
                            pre_data[i],
                            test_n
                        ))
                except:
                    continue
            #5 & 0.568 & 0.301 & 0.584 & 0.321  & 0.594 & 0.325 & 0.596 & 0.330\\




#\begin{tabular}{|c||c|c|c|c|c|c|c|c|}\hline
#    & \multicolumn{2}{c|}{Linear} & \multicolumn{2}{c|}{Dense} & \multicolumn{2}{c|}{GRU} & \multicolumn{2}{c|}{Bid-GRU} \\\cline{1-9}
#    Days & AUC & Pre & AUC & Pre & AUC & Pre & AUC & Pre \\\hline
#    5 & 0.568 & 0.301 & 0.584 & 0.321  & 0.594 & 0.325 & 0.596 & 0.330\\
#    6 & 0.599 & 0.345 & 0.584 & 0.332  & 0.591 & 0.336 & 0.600 & 0.343\\
#    7 & 0.598 & 0.354 & 0.589 & 0.351  & 0.593 & 0.349 & 0.598 & 0.361\\
#    8 & 0.592 & 0.361 & 0.599 & 0.369  & 0.604 & 0.381 & 0.602 & 0.371\\
#    9 & 0.609 & 0.391 & 0.606 & 0.392  & 0.604 & 0.391 & 0.616 & 0.396\\
#    10 & 0.610 & 0.402 & 0.613 & 0.410  & 0.620 & 0.417 & 0.613 & 0.412\\
#    11 & 0.614 & 0.419 & 0.621 & 0.442  & 0.633 & 0.454 & 0.625 & 0.436\\
#    12 & 0.625 & 0.451 & 0.629 & 0.462  & 0.642 & 0.476 & 0.640 & 0.469\\
#\hline
#\end{tabular}