import os
import matplotlib.pyplot as plt
import matplotlib.font_manager
from operator import add

dataroot = 'result_topk'
dataname = ['nr','gpcr','ic']
fold = list(range(5,21))


## ----------------------------- copy K_values out -----------------------------------
#for dataset in dataname:
#  for foldid in fold:
#   for fd in range(1, foldid+1):
#     path = dataroot + '/' + dataset + '/' + str(foldid) + '/fold' + str(fd) + '/'
#     cmd = 'cp '+path+'K_values.txt '+path+'../'
#     os.system(cmd)


## ------------------------------ obtain avg kronls topk ------------------------------
#for dataset in dataname:
#  for foldid in fold:
#    f = open(dataroot+'/'+dataset+'/'+str(foldid)+'/K_values.txt','r')
#    kvalues = f.read()
#    kvalues = kvalues.split(' ')
#    kvalues = kvalues[:-1]
#    K_set = []
#    for e in kvalues:
#      K_set.append(int(e))      
#    f.close()
#    topK_avg = [0]*len(K_set)
#    f = open(dataroot + '/' + dataset + '/' + str(foldid) + '/'+'kronrls_topK_avg.txt','w')
#    for fd in range(1, foldid+1):
#       path = dataroot + '/' + dataset + '/' + str(foldid) + '/fold' + str(fd) + '/'
#       f2 = open(path+'kronrls_topK.txt','r')
#       tmp = []
#       for row in f2:
#         tmp.append(float(row))
#       f2.close()
#       topK_avg = map(add, topK_avg, tmp)
#    for e in topK_avg:
#       e = e/foldid
#       f.write(str(e)+'\n')
#    f.close()
    

# ----------------------------- plot best mean across folds curve --------------------------------
for dataset in dataname:
  bsf_rate = []
  bsf_mean = 10000
  bsf_foldnum = 0
  for foldid in fold:
    path = dataroot + '/' + dataset + '/' + str(foldid) + '/'
    rate = []
    f = open(path + 'kronrls_topK_avg.txt', 'r')
    for e in f:
      rate.append(float(e))
    if sum(rate)/len(rate) < bsf_mean:
      bsf_mean = sum(rate)/len(rate)
      bsf_rate = rate
      bsf_foldnum = foldid
    f.close()
  path = dataroot + '/' + dataset + '/' + str(bsf_foldnum) + '/'
  f = open(path+ 'K_values.txt', 'r')
  kvalues = f.read()
  kvalues = kvalues.split(' ')
  kvalues = kvalues[:-1]
  K_set = []
  for e in kvalues:
    K_set.append(int(e))      
  f.close()
  plt.clf()
  plt.plot(K_set, bsf_rate)
  plt.xlabel('Top Size')
  plt.ylabel('Capture Percentage')
  #plt.ylim([0.0, 1.05])
  #plt.xlim([0.0, 1.0])
  plt.title('Capture Rate in Top Size in dataset '+dataset+' with '+str(bsf_foldnum)+' folds')
  #plt.legend(loc="lower left")
  plt.show()

   
