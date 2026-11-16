"""Resource-aware complete-mechanism experiments. No query-based model selection."""
import argparse, copy, os, random, time
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from common import ROOT,read,save,torch_save,fingerprint,load_data,predict
from metrics import classification,effort,chosen_threshold

class LR(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.linear=nn.Linear(dim,2); nn.init.normal_(self.linear.weight,std=.01)
    def forward(self,x): return self.linear(x)

def oversample(x,y,rng):
    counts=np.bincount(y,minlength=2); idx=list(range(len(y)))
    for label in [0,1]:
        extra=int(max(counts)-counts[label])
        if extra:idx.extend(rng.choice(np.flatnonzero(y==label),extra,replace=True).tolist())
    return torch.from_numpy(x[idx]),torch.from_numpy(y[idx])

def cosine_weights(clients,public):
    # Same per-record cosine and zero-sum rule as the visible author code.
    p=torch.nn.functional.normalize(torch.from_numpy(public),dim=1,eps=1e-8)
    rows=[]
    for c in clients.values():
        x=torch.nn.functional.normalize(torch.from_numpy(c['x']),dim=1,eps=1e-8)
        rows.append((p @ x.T).mean(dim=1))
    density=torch.stack(rows);den=density.sum(dim=0)
    alpha=torch.zeros_like(density); valid=den!=0;alpha[:,valid]=density[:,valid]/den[valid]
    assert torch.isfinite(alpha).all()
    return alpha

def fed_round(model,clients,public,alpha,cfg,rng):
    ids=rng.choice(len(public),min(cfg['public_sample_limit'],len(public)),replace=False)
    px=torch.from_numpy(public[ids]); states=[]; logits=[]; counts=[]; local_losses=[]
    for c in clients.values():
        local=copy.deepcopy(model); local.train(); opt=torch.optim.AdamW(local.parameters(),lr=cfg['local_lr'])
        sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',patience=3,factor=.2)
        x,y=c['sampled']; loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=cfg['batch_size'],shuffle=True)
        for _ in range(cfg['local_epochs']):
            correct=0; total_loss=0.
            for bx,by in loader:
                opt.zero_grad(); z=local(bx); loss=nn.functional.cross_entropy(z,by);loss.backward();opt.step()
                correct+=int((z.argmax(1)==by).sum());total_loss+=float(loss.detach())*len(by)
            sched.step(correct/len(y))
        local_losses.append(total_loss/len(y)); states.append(copy.deepcopy(local.state_dict()));counts.append(len(c['y']))
        local.eval()
        with torch.no_grad(): logits.append(local(px))
    state=OrderedDict((k,sum(s[k]*(n/sum(counts)) for s,n in zip(states,counts))) for k in states[0])
    model.load_state_dict(state); model.train()
    teacher=(torch.stack(logits)*alpha[:,ids,None]).sum(0).detach()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['distill_lr'])
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',patience=1,factor=.2)
    temp=cfg['temperature']; epoch_loss=0.
    for _ in range(cfg['distill_epochs']):
        losses=[]
        for start in range(0,len(px),cfg['batch_size']):
            bx=px[start:start+cfg['batch_size']];tz=teacher[start:start+cfg['batch_size']]
            opt.zero_grad();z=model(bx)
            # Explicit mean matches author's KLDivLoss default (batch AND class mean).
            loss=temp**2*nn.functional.kl_div(nn.functional.log_softmax(z/temp,dim=1),nn.functional.softmax(tz/temp,dim=1),reduction='none').mean()
            loss.backward();opt.step();losses.append(float(loss.detach()))
        epoch_loss=float(np.mean(losses));sched.step(epoch_loss)
    return dict(local_loss=float(np.mean(local_losses)),kd_loss=epoch_loss,public_indices=ids.tolist(),alpha_min=float(alpha[:,ids].min()),alpha_max=float(alpha[:,ids].max()))

def disjoint_batches(project,config,rng,device):
    # Native sampler's replacement path can put identical rows in support and query.
    # Split the available row identities first, then sample independently from each.
    n=len(project.y_train);b=config['training']['batch_size']; assert n>=2
    indices=rng.permutation(n);split=max(1,min(n-1,n//2));left,right=indices[:split],indices[split:]
    si=rng.choice(left,b,replace=len(left)<b);qi=rng.choice(right,b,replace=len(right)<b)
    assert not set(si)&set(qi)
    return (torch.as_tensor(project.x_train[si],device=device),torch.as_tensor(project.y_train[si],device=device),
            torch.as_tensor(project.x_train[qi],device=device),torch.as_tensor(project.y_train[qi],device=device))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target',required=True);ap.add_argument('--seed',required=True,type=int)
    ap.add_argument('--rounds',type=int);ap.add_argument('--out',default='runs');ap.add_argument('--stop-after',type=int)
    args=ap.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    cfg,task,clients,target,public,audit=load_data(args.target); rounds=args.rounds or cfg['rounds']
    assert args.out!='runs' or rounds==cfg['rounds']; assert args.seed in cfg['seeds']
    folder=ROOT/args.out/args.target/str(args.seed);folder.mkdir(parents=True,exist_ok=True)
    code_hash=fingerprint(); identity={'fingerprint':code_hash,'target':args.target,'seed':args.seed,'rounds':rounds}
    if (folder/'metrics.json').exists():
        assert read(folder/'metrics.json')['identity']==identity;print('already completed',flush=True);return
    save(folder/'data_audit.json',audit); dim=target['x'].shape[1]; optimizer=None
    if cfg['method']=='ecfml':
        from ecfml_vendor import training as ec
        from ecfml_vendor.model import build_model
        from ecfml_vendor.preprocessing import PreparedProject
        from ecfml_vendor.differential_privacy import privatize_updates
        from ecfml_vendor.utils import average_tensor_dicts
        ec.support_query_batch=disjoint_batches
        config=read(ROOT/'ecfml_config.json');config['model']['input_dim']=dim;config['training']['seed']=args.seed
        projects={n:PreparedProject(n,c['x'],c['y'],c['vx'],c['vy']) for n,c in clients.items()}
        tp=PreparedProject(args.target,target['support_x'],target['support_y'],target['x'],target['y'])
        model=build_model(config);optimizer=torch.optim.Adam(model.parameters(),lr=config['training']['learning_rate'])
        save(folder/'resolved_model_config.json',config)
    else:
        model=LR(dim)
        for c in clients.values():c['sampled']=oversample(c['x'],c['y'],rng)
        alpha=cosine_weights(clients,public)
    history=[]; start=0
    if (folder/'checkpoint.pt').exists():
        state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert state['identity']==identity
        model.load_state_dict(state['model']);start=state['round'];history=state['history']
        if optimizer:optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);rng.bit_generator.state=state['numpy_rng'];random.setstate(state['python_rng'])
    for r in range(start,rounds):
        before=time.time()
        if cfg['method']=='ecfml':
            updates=[];losses=[]
            for project in projects.values():
                update,loss=ec.first_order_client_update(model,project,config,rng,torch.device('cpu'))
                updates.append(privatize_updates(update,clipping_bound=config['privacy']['clipping_bound'],epsilon=config['privacy']['epsilon'],enabled=True,clipping_mode='per_coordinate'));losses.append(loss)
            avg=average_tensor_dicts(updates);optimizer.zero_grad(set_to_none=True)
            for name,p in model.named_parameters():p.grad=avg[name]
            optimizer.step();record={'meta_query_loss':float(np.mean(losses))}
        else:record=fed_round(model,clients,public,alpha,cfg,rng)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        record.update(round=r+1,seconds=time.time()-before);history.append(record)
        state=dict(identity=identity,round=r+1,model=model.state_dict(),optimizer=optimizer.state_dict() if optimizer else None,
                   history=history,torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,python_rng=random.getstate())
        torch_save(folder/'checkpoint.pt',state)
        if cfg['method']=='feddp' and r>=rounds-10:torch_save(folder/f'round_{r+1}.pt',model.state_dict())
        save(folder/'progress.json',dict(round=r+1,total=rounds,updated_unix=time.time(),identity=identity))
        print(f'{args.target} seed={args.seed} round={r+1}/{rounds} seconds={record["seconds"]:.3f}',flush=True)
        if args.stop_after and r+1>=args.stop_after:return
    if cfg['method']=='ecfml':model=ec.adapt_to_target(model,tp,config,torch.device('cpu'))
    torch_save(folder/'final_model.pt',model.state_dict())
    values=[];val_frames=[]
    for name,c in clients.items():
        p=predict(model,c['vx']);values.append((c['vy'],p))
        val_frames.append(pd.DataFrame(dict(project=name,row=c['validation'],y=c['vy'],probability=p)))
    threshold=chosen_threshold(values);p=predict(model,target['x']);y=target['y']
    loc=target['loc'] if target['effort_valid'] else np.full(len(y),np.nan)
    pd.concat(val_frames).to_csv(folder/'validation_predictions.csv',index=False)
    pd.DataFrame(dict(row=target['query'],y=y,probability=p,loc=target['loc'])).to_csv(folder/'predictions.csv',index=False)
    result=dict(identity=identity,method=cfg['method'],family=task['family'],target=args.target,seed=args.seed,rounds=rounds,threshold=threshold,
                metrics={**classification(y,p,threshold),**effort(y,p,loc)},native_threshold_metrics=classification(y,p,.5),
                test_n=len(y),support_n=len(target['support']),private_clients=len(clients),public_n=len(public) if public is not None else 0,
                effort_valid=target['effort_valid'],completed_unix=time.time(),train_seconds=sum(h['seconds'] for h in history))
    if cfg['method']=='feddp':
        last=[]
        for r in range(max(1,rounds-9),rounds+1):
            model.load_state_dict(torch.load(folder/f'round_{r}.pt',weights_only=True));rp=predict(model,target['x'])
            last.append(dict(round=r,**classification(y,rp,.5)))
            pd.DataFrame(dict(row=target['query'],y=y,probability=rp)).to_csv(folder/f'predictions_round_{r}.csv',index=False)
        save(folder/'last10_native_metrics.json',last)
        result['last10_native_mean']={k:float(np.mean([v[k] for v in last])) for k in last[0] if k!='round'}
    save(folder/'history.json',history);save(folder/'metrics.json',result)

if __name__=='__main__': main()
