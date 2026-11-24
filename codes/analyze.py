import csv, itertools, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
C=['Language','Ideas','Cohesion','Academic']; R=['Human1','Human2','Model']; pairs=[(0,1),(2,0),(2,1)]
def read(p):
 with (P/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,header,rows):
 with (P/p).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(header.split());w.writerows(rows)
m=read('model_ratings.csv'); I=sorted({r['essay_id'] for r in read('LICA_LEAF_100_descriptives.csv')}); assert len(I)==100
d=defaultdict(set); seen=set(); exclusions=[]
for r in m:
 k=(r['essay_id'],r['criterion']); assert k not in seen and r['actual_model_id']=='qwen2.5:7b' and r['provider']=='ollama';seen.add(k);d[*k,'Model'].add(int(r['score']))
assert seen==set(itertools.product(I,C))
for r in read('ratings_transcribed.csv'):
 if r['essay_id'] in I:d[r['essay_id'],r['criterion'],r['rater_id']].add(int(r['score']))
X=np.full((100,4,3),np.nan)
for i,c,r in itertools.product(range(100),range(4),range(3)):
 s=d[I[i],C[c],R[r]]; assert s<=set(range(1,6))
 if len(s)==1:X[i,c,r]=next(iter(s))
 else:exclusions.append([I[i],C[c],R[r],'conflict' if s else 'missing','|'.join(map(str,sorted(s)))])
save('excluded.csv','essay_id criterion rater reason recorded_scores',exclusions)
assert len(exclusions)==42 and all(e[2:4]==['Human2','conflict'] for e in exclusions)
complete=np.isfinite(X).all(axis=(1,2)); save('complete_ids.csv','essay_id',[[I[i]] for i in np.where(complete)[0]])
def metric(a,b):
 d=a-b; var=a.var(axis=-1)+b.var(axis=-1)+d.mean(axis=-1)**2; mse=(d*d).mean(axis=-1)
 return [1-np.divide(mse,var,out=np.full_like(var,np.nan),where=var>0),abs(d).mean(axis=-1),(a==b).mean(axis=-1)*100,(abs(d)<=1).mean(axis=-1)*100,d.mean(axis=-1),np.sqrt(mse)]
def ci(v):
 v=v[np.isfinite(v)]; return np.quantile(v,[.025,.975]) if len(v) else [np.nan,np.nan]
S=[]; D=[]; M=[]; U=[]; F=[]; L=[]
prior={r['criterion']:r for r in read('prior_human_summary.csv')}
for mode in ('pairwise','matched','complete'):
 for c,crit in enumerate(C):
  boots=[]; qs=[]
  for a,b in pairs:
   mask=np.isfinite(X[:,c,[a,b]]).all(axis=1) if mode=='pairwise' else (np.isfinite(X[:,c,:]).all(axis=1) if mode=='matched' else complete)
   x,y=X[mask,c,:][:,[a,b]].T;n=len(x); j=np.random.default_rng(42).integers(n,size=(5000,n)); est=metric(x,y); boot=metric(x[j],y[j]); boots.append(boot[0]);qs.append(est[0]); label=R[a]+'-'+R[b]
   if mode=='pairwise' and (a,b)==(0,1):assert n==int(prior[crit]['n']) and np.isclose(est[0],float(prior[crit]['qwk']))
   S.append([mode,crit,label,n,*[v for e,s in zip(est,boot) for v in [e,*ci(s)]],np.isfinite(boot[0]).sum(),x.mean(),y.mean(),x.std(ddof=1),y.std(ddof=1),int(sum(x<y)),int(sum(x==y)),int(sum(x>y))]);
   if mode=='pairwise':
    U.extend([crit,label,I[i],int(v),int(w)] for i,v,w in zip(np.where(mask)[0],x,y));
    M.extend([crit,label,k,l,int(sum((x==k)&(y==l)))] for k,l in itertools.product(range(1,6),repeat=2));
    d=x-y; sd=d.std(ddof=1); L.append([crit,label,n,d.mean(),sd,d.mean()-1.96*sd,d.mean()+1.96*sd])
  if mode!='pairwise':
   for k in (1,2):
    diff=boots[k]-boots[0];D.append([mode,crit,k,n,qs[k]-qs[0],*ci(diff),np.isfinite(diff).sum()])
for c,r,k in itertools.product(range(4),range(3),range(1,6)):
 x=X[:,c,r];F.append([C[c],R[r],k,int(sum(x==k)),int(np.isfinite(x).sum())])
save('agreement.csv','subset criterion pair n '+' '.join(f'{k} {k}_lo {k}_hi' for k in ['qwk','mae','exact_pct','within_one_pct','bias','rmse'])+' valid_qwk_boots mean_a mean_b sd_a sd_b a_below_b a_equal_b a_above_b',S)
save('qwk_contrasts.csv','subset criterion model_human_number n delta_vs_human_human lo hi valid_boots',D)
save('score_frequencies.csv','criterion rater score count valid_n',F)
save('confusion_matrices.csv','criterion pair score_a score_b count',M)
save('pairs_used.csv','criterion pair essay_id score_a score_b',U)
save('bland_altman_descriptive.csv','criterion pair n bias sd_difference lower_loa upper_loa',L)
fig,ax=plt.subplots(figsize=(8,5),layout='constrained'); colors=['#222222','#0571b0','#ca562c']
for ix,(a,b) in enumerate(pairs):
 label=R[a]+'-'+R[b]; s=[r for r in S if r[0]=='matched' and r[2]==label]; v=np.array([r[4:7] for r in s]);ax.errorbar(v[:,0],np.arange(4)+(ix-1)*.20,xerr=np.array([v[:,0]-v[:,1],v[:,2]-v[:,0]]),fmt='o',capsize=3,label=label,color=colors[ix]);
ax.axvline(0,color='grey',ls=':');ax.set_yticks(range(4),C);ax.invert_yaxis();ax.set(xlim=(-.35,1.02),xlabel='Quadratic-weighted kappa (95% bootstrap CI)',title='Agreement on identical essays within each criterion');ax.legend(loc='lower right');
for ext in ['png','pdf']:fig.savefig(P/f'qwk_matched.{ext}',dpi=250)
print('complete',sum(complete));
for r in S:
 if r[0]!='complete': print(*r[:4],*[round(x,3) for x in r[4:19]])
print('contrasts',D)
