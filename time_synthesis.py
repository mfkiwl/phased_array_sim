import pandas as pd
import numpy as np

'''
time is ordered by:
ELO
GA
BPSO.

Example csv content:
Algorithm,Average Time (ms)
ELO,0.16314029693603516
GA,3627.778687477112
BPSO,4736.615505218506
'''

filenames = [f"ult_eval_data\\average_times_{i}.csv" for i in range(1, 3)]
dfs = [pd.read_csv(filename) for filename in filenames]
# extract the times for ELO, GA and BPSO, as a numpy array
times = np.array([df['Average Time (ms)'].values for df in dfs])
# Calculate the average time for each algorithm
average_times = np.mean(times, axis=0)
# print the average times
print("Average Times (ms):")
for i, algorithm in enumerate(dfs[0]['Algorithm']):
    print(f"{algorithm}: {average_times[i]:.2f} ms")

# save the average times to a new csv file
average_times_df = pd.DataFrame({
    'Algorithm': dfs[0]['Algorithm'],
    'Average Time (ms)': average_times
})
average_times_df.to_csv('ult_eval_data\\average_times_synthesized.csv', index=False)
