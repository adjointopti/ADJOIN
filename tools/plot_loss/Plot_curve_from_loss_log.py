import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys

# plt.style.use('science')

loss_log_file = sys.argv[1]  #'hwgen_alter_epoch/yosee/loss_log.txt'
save_path = os.path.split(loss_log_file)[0]
lines = [l.strip() for l in open(loss_log_file)]
common_loss = []
for l in lines:
    if 'step:' in l:
        # pattern = re.compile(r"(?<='Common_Loss': tensor\()\d+\.?\d*")
        # loss = pattern.findall(l)
        loss = re.findall(r"(?<='Common_Loss': tensor\()\d+\.?\d*", l)
        loss = np.array(loss[0]).astype('float32')
        common_loss.append(loss)
step_num = re.findall(r"(?<=step:)\d*", lines[-1])
step_num = np.array(step_num[0]).astype('int32')
steps = range(1, step_num+1)
converage_value = np.mean(common_loss[-50:-1])

plt.figure()
plt.plot(steps,common_loss)
plt.plot([0,step_num],[converage_value,converage_value], 'r-',lw=1,dashes=[2, 2])
plt.axhline()
plt.xlim((0, step_num))
plt.ylim((0, 0.4))
plt.tick_params(labelsize=12)
font1 = {'family': 'Times New Roman',
'weight': 'normal',
'size': 14,}
plt.xlabel('Steps', font1)
plt.ylabel('Common Loss', font1)
plt.savefig(os.path.join(save_path,'CommonLoss.png'), dpi=1000)
# plt.show()
