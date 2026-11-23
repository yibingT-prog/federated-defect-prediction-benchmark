import json
from pathlib import Path
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parents[4]/'results/fedcpdp-privacy-utility-20261120/report';a=json.loads((O/'analysis.json').read_text())
cs=a['conditions'];labels=['No perturbation','Clip only','0.1','0.25','0.5','1.0']
fig,axs=plt.subplots(2,2,figsize=(9,4.3),layout='constrained')
for ax,k in zip(axs.flat,['ROC_AUC','MCC','Recall','G_mean']):
    ax.plot(range(6),[a['overall'][c][k]['mean'] for c in cs],marker='o',color='#225b88',lw=1.8)
    ax.set_xticks(range(6),labels,fontsize=8);ax.set_title(k.replace('_',' '),fontsize=11);ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
fig.savefig(O/'utility.png',dpi=220);fig.savefig(O/'utility.svg');plt.close(fig)
