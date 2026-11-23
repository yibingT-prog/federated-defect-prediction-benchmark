import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, f1_score

def classification(y, p, t):
    tn,fp,fn,tp=confusion_matrix(y,p>=t,labels=[0,1]).ravel()
    return dict(ROC_AUC=float(roc_auc_score(y,p)),MCC=float(matthews_corrcoef(y,p>=t)),Recall=float(tp/(tp+fn)),G_mean=float(np.sqrt(tp/(tp+fn)*tn/(tn+fp))),F1=float(f1_score(y,p>=t,zero_division=0)))

def effort(y,p,loc):
    if not np.isfinite(loc).all() or (loc<=0).any() or y.sum()==0:
        return {f'{m}@{b}':None for m in ['CostEffect','Popt'] for b in [5,10,20,40]}
    ids=np.arange(len(y)); result={}
    def area(order,b):
        x=np.r_[0,np.cumsum(loc[order])/loc.sum()];z=np.r_[0,np.cumsum(y[order])/y.sum()]
        xx=np.r_[x[x<b],b];zz=np.interp(xx,x,z)
        return float(zz[-1]),float(np.trapezoid(zz,xx))
    for b in [5,10,20,40]:
        ce,a=area(np.lexsort((ids,-p/loc)),b/100)
        _,best=area(np.lexsort((ids,-y/loc)),b/100);_,worst=area(np.lexsort((ids,y/loc)),b/100)
        result[f'CostEffect@{b}']=ce;result[f'Popt@{b}']=(a-worst)/(best-worst) if abs(best-worst)>1e-12 else None
    return result

def chosen_threshold(vals):
    choices=[]
    for t in np.linspace(.05,.95,37):
        mm=[];rr=[]
        for y,p in vals:
            z=p>=t;tp=int(np.sum(z&(y==1)));fp=int(np.sum(z&(y==0)));fn=int(np.sum(~z&(y==1)));tn=int(np.sum(~z&(y==0)))
            den=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
            mm.append((tp*tn-fp*fn)/den if den else 0.);rr.append(tp/(tp+fn))
        choices.append(((np.mean(mm),np.mean(rr),-t),float(t)))
    return max(choices)[1]
