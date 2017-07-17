import matplotlib.pyplot as plt
import numpy as np

n_groups = 18

nr_means = [0.4630,0.5187,0.0709,0.0685,0.2041,0.4321,0.4078,0.5368,0.1989,0.1870,0.2310,0.2195,0.1830,0.0946,0.1702,0.4477,0.3293,0.6200]
gpcr_means = [0.32,0.6188,0.0633,0.0519,0.4960,0.6208,0.6213,0.6440,0.2183,0.2169,0.3763,0.3841,0.2625,0.1230,0.2613,0.5324,0.4240,0.6700]
ic_means = [0.6789,0.8679,0.1169,0.1106,0.7671,0.8553,0.8693,0.8769,0.3088,0.3187,0.5359,0.5560,0.5133,0.1608,0.5474,0.7505,0.7116,0.9000]

fig, ax = plt.subplots(figsize = (30,5))

index = np.arange(n_groups)
bar_width = 0.20

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, nr_means, bar_width,
                 alpha=opacity,
                 color='b',                 
                 error_kw=error_config,
                 label='nr')

rects2 = plt.bar(index + bar_width, gpcr_means, bar_width,
                 alpha=opacity,
                 color='r',                 
                 error_kw=error_config,
                 label='gpcr')

rects3 = plt.bar(index + 2*bar_width, ic_means, bar_width,
                 alpha=opacity,
                 color='g',                 
                 error_kw=error_config,
                 label='ic')


plt.xlabel('Group')
plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(index + bar_width / 2, ('PPI-GIP', 'SW-GIP', 'BLM-KA', 'BLM-MEAN','KBMF2MKL', 'KRONRLS-KA', 'KRONRLS-MEAN', 'KRONRLS-MKL','LAPRLS-KA', 'LAPRLS-MEAN', 'NRWRH-KA', 'NRWRH-MEAN','PKM-KA', 'PKM-MAX', 'PKM-MEAN', 'SITAR', 'WANG-MKL','KRONRLSMKL-STACK'))
plt.legend()

plt.tight_layout()
plt.figure(figsize=(300,400))
plt.show()