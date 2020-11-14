import os
import pandas as pd

exp = 'hyper_lrt_sched' #CHANGE EXPNAME
results_path = './results/results-{}'.format(exp)
outpath = os.path.join(results_path, 'results-{}.csv'.format(exp))
results = os.listdir(results_path)
compiled_results = []
for exp in results:
    if os.path.splitext(exp)[-1] != ".csv":
        continue
    csvfile = os.path.join(results_path, exp)
    df = pd.read_csv(csvfile)
    labels = df.columns
    data = df.to_numpy()
    compiled_results.append(data[-1])

compiled_df = pd.DataFrame(data=compiled_results, columns=labels)
compiled_df.to_csv(outpath, index=False)
print(compiled_df)
