import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

folder = 'ult_eval_data'
filename = 'ga_elo_bpso_compare_.csv'
df = pd.read_csv(f'{folder}\\{filename}')

average_df = df.drop(columns=['Unnamed: 0'])  # Drop the index column

stuck_bits = np.arange(0, 33)  # Assuming stuck bits are from 0 to 1000 in increments of 100
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
fig, axs = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1, 1]})
fig.suptitle(f"Comparing ELO, GA and BPSO")
fig.subplots_adjust(hspace=0.4)  # Adjust space between subplots
#axs[0].plot(stuck_bits, avg_nse_quantised_list, label='Quantised', color='blue')
axs[0].plot(stuck_bits, avg_isll_broken_list, label='Broken', color='red')
axs[0].plot(stuck_bits, avg_isll_elo_list, label='ELO', color='green')
axs[0].plot(stuck_bits, avg_isll_ga_list, label='GA', color='purple')
axs[0].plot(stuck_bits, avg_isll_bpso_list, label='BPSO', color='cyan')
#axs[0].set_title('Total normalised_SE')
#axs[0].set_xlabel('Proportion of Stuck Bits')
axs[0].set_ylabel('ISLL [dB]')
axs[0].legend()
axs[0].grid()
#axs[1].plot(stuck_bits, avg_mb_nse_quantised_list, label='Quantised', color='blue')
axs[1].plot(stuck_bits, avg_pssl_broken_list, label='Broken', color='red')
axs[1].plot(stuck_bits, avg_pssl_elo_list, label='ELO', color='green')
axs[1].plot(stuck_bits, avg_pssl_ga_list, label='GA', color='purple')
axs[1].plot(stuck_bits, avg_pssl_bpso_list, label='BPSO', color='cyan')
#axs[1].set_title('Main Beam PSSL')
#axs[1].set_xlabel('Proportion of Stuck Bits')
axs[1].set_ylabel('PSLL [dB]')
axs[1].legend()
axs[1].grid()
axs[2].plot(stuck_bits, avg_pbp_broken_list, label='Broken', color='red')
axs[2].plot(stuck_bits, avg_pbp_elo_list, label='ELO', color='green')
axs[2].plot(stuck_bits, avg_pbp_ga_list, label='GA', color='purple')
axs[2].plot(stuck_bits, avg_pbp_bpso_list, label='BPSO', color='cyan')
#axs[2].set_title('PBP')
axs[2].set_xlabel('Proportion of Stuck Bits')
axs[2].set_ylabel('PBP [dB]')
axs[2].legend()
axs[2].grid()
# Save the figure
#fig.savefig(output_file_name, dpi=300, bbox_inches='tight')
plt.show()