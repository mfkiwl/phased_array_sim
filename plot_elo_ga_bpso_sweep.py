import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
1-10: old files, 11-18: new files

There are several files to read, each containing the results of 100 trials. They need to be combined with a simple average. 
The files are named as follows:
folder: ult_eval_data
names: 
- 'ga_elo_bpso_compare_1.csv'
- 'ga_elo_bpso_compare_2.csv'
- 'ga_elo_bpso_compare_3.csv'
- ...
- 'ga_elo_bpso_compare_10.csv'

Each file has an initial index column, with no header. The rest of the columns are:
avg_nse_broken,avg_nse_elo,avg_nse_ga,avg_nse_bpso,avg_pssl_broken,avg_pssl_elo,avg_pssl_ga,avg_pssl_bpso,avg_pbp_broken,avg_pbp_elo,avg_pbp_ga,avg_pbp_bpso
'''
folder = 'ult_eval_msb'
file_names = [f'ga_elo_bpso_compare_{i}.csv' for i in range(1, 9)]
output_file_name = 'visuals\\ga_elo_bpso_compare_average_msb.png'

data_frames = []
for file_name in file_names:
    file_path = f'{folder}\\{file_name}'
    df = pd.read_csv(file_path)
    data_frames.append(df)
# Combine all data frames into one
average_df = sum(data_frames) / len(data_frames)

stuck_bits = average_df["Unnamed: 0"].to_numpy()
total_bits = stuck_bits[-1]
stuck_bits = stuck_bits / total_bits  # Normalise stuck bits to be between 0 and 1
avg_nse_broken_list = average_df['avg_nse_broken'].to_numpy()
avg_nse_elo_list = average_df['avg_nse_elo'].to_numpy()
avg_nse_ga_list = average_df['avg_nse_ga'].to_numpy()
avg_nse_bpso_list = average_df['avg_nse_bpso'].to_numpy()
avg_pssl_broken_list = average_df['avg_pssl_broken'].to_numpy()
avg_pssl_elo_list = average_df['avg_pssl_elo'].to_numpy()
avg_pssl_ga_list = average_df['avg_pssl_ga'].to_numpy()
avg_pssl_bpso_list = average_df['avg_pssl_bpso'].to_numpy()
avg_pbp_broken_list = average_df['avg_pbp_broken'].to_numpy()
avg_pbp_elo_list = average_df['avg_pbp_elo'].to_numpy()
avg_pbp_ga_list = average_df['avg_pbp_ga'].to_numpy()
avg_pbp_bpso_list = average_df['avg_pbp_bpso'].to_numpy()
avg_isll_broken_list = average_df['avg_isll_broken'].to_numpy()
avg_isll_elo_list = average_df['avg_isll_elo'].to_numpy()
avg_isll_ga_list = average_df['avg_isll_ga'].to_numpy()
avg_isll_bpso_list = average_df['avg_isll_bpso'].to_numpy()

# 3 subplots, one for NSe, one for PSSL and one for PBP
fig, axs = plt.subplots(4, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [1, 1, 1, 1]})
fig.suptitle(f"Comparing ELO, GA and BPSO")
fig.subplots_adjust(hspace=0.4)  # Adjust space between subplots
#axs[0].plot(stuck_bits, avg_nse_quantised_list, label='Quantised', color='blue')
axs[0].plot(stuck_bits, avg_nse_broken_list, label='Broken', color='red')
axs[0].plot(stuck_bits, avg_nse_elo_list, label='ELO', color='blue')
axs[0].plot(stuck_bits, avg_nse_ga_list, label='GA', color='magenta')
axs[0].plot(stuck_bits, avg_nse_bpso_list, label='BPSO', color='green')
#axs[0].set_title('Total normalised_SE')
#axs[0].set_xlabel('Proportion of Stuck Bits')
axs[0].set_ylabel('$MSE_{norm}$')
axs[0].legend()
axs[0].grid()
#axs[1].plot(stuck_bits, avg_mb_nse_quantised_list, label='Quantised', color='blue')
axs[1].plot(stuck_bits, avg_pssl_broken_list, label='Broken', color='red')
axs[1].plot(stuck_bits, avg_pssl_elo_list, label='ELO', color='blue')
axs[1].plot(stuck_bits, avg_pssl_ga_list, label='GA', color='magenta')
axs[1].plot(stuck_bits, avg_pssl_bpso_list, label='BPSO', color='green')
#axs[1].set_title('Main Beam PSSL')
#axs[1].set_xlabel('Proportion of Stuck Bits')
axs[1].set_ylabel('PSLL [dB]')
axs[1].legend()
axs[1].grid()
#axs[2].plot(stuck_bits, avg_mb_nse_quantised_list, label='Quantised', color='blue')
axs[2].plot(stuck_bits, avg_isll_broken_list, label='Broken', color='red')
axs[2].plot(stuck_bits, avg_isll_elo_list, label='ELO', color='blue')
axs[2].plot(stuck_bits, avg_isll_ga_list, label='GA', color='magenta')
axs[2].plot(stuck_bits, avg_isll_bpso_list, label='BPSO', color='green')
#axs[2].set_title('Main Beam islL')
#axs[2].set_xlabel('Proportion of Stuck Bits')
axs[2].set_ylabel('ISLL [dB]')
axs[2].legend()
axs[2].grid()
axs[3].plot(stuck_bits, avg_pbp_broken_list, label='Broken', color='red')
axs[3].plot(stuck_bits, avg_pbp_elo_list, label='ELO', color='blue')
axs[3].plot(stuck_bits, avg_pbp_ga_list, label='GA', color='magenta')
axs[3].plot(stuck_bits, avg_pbp_bpso_list, label='BPSO', color='green')
#axs[3].set_title('PBP')
axs[3].set_xlabel('Proportion of Stuck Bits')
axs[3].set_ylabel('PBP [dB]')
axs[3].legend()
axs[3].grid()
# Save the figure
fig.savefig(output_file_name, dpi=300, bbox_inches='tight')
plt.show()