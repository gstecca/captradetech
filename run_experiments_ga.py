# Copyright (c) 2025 Massimiliano Caramia, Anna Laura Pala, Giuseppe Stecca
# Authors: Massimiliano Caramia, Anna Laura Pala, Giuseppe Stecca
# Email: giuseppe.stecca@gmail.com

import ga as g
import json
import pandas as pd
import random

DEBUG = False
RESET = True
random.seed(0)
with open('PARAMS_GA.json', 'r') as f:
        params = json.load(f)

params['algotype'] = 'GA'
params["folder_input"] = "instances_IB"  
params["folder_output"] = "results_IB_ga"
params["time_limit"] = 100  # seconds
params['maxIterations'] = 30
 
instance_file = 'instances_bilevel_IB_GA.csv'
#instance_file = "instances_bilevel_IR_GA_tuning.csv"
#id = 'I_0V'
#deltas = [0.7, 0.75, 0.80, 0.85, 0.90]
df = pd.read_csv(instance_file, sep=',', index_col='id')#, dtype={'himprov': float})
if RESET:
    df['lobjval'] = -1.0
    df['lz1'] = -1.0
    df['lz2'] = -1.0
    df['fobjval'] = -1.0
    df['tot_runtime'] = -1.0
    df['gapf'] = -1.0
    df['best_it'] = -1.0
    df['himprov'] = -1.0
    df['solved'] = False
#if id != None:
#        df = df.loc[[id]]


for index, row in df.iterrows():
        id = index
        #for delta in deltas:
        delta = row['delta']

        params['instancename'] = id
        params['delta'] = delta
        params['params_from_params_file'] = False
        if row['solved']:
                continue
        print(f"running instance {id} with delta {delta}")
        result = g.run(params)
        df.at[index, 'solved'] = True
        df.at[index, 'lobjval'] = round(result['lobjval'],2)
        df.at[index, 'lz1'] = round(result['lz1'],2)
        df.at[index, 'lz2'] = round(result['lz2'],2)
        df.at[index, 'fobjval'] = round(result['fobjval'],2)
        df.at[index, 'tot_runtime'] = round(result['tot_runtime'],2)
        df.at[index, 'gapf'] = round(result['gapf'],2)
        df.at[index, 'best_it'] = result['best_it']
        df.to_csv(instance_file)
        print(f"finished instance {id} with delta {delta}")
        if DEBUG:
            break
df.to_csv(instance_file)
