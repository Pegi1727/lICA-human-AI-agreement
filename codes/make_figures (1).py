import argparse
import csv
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

C = ['Language','Ideas','Cohesion','Academic']
NAMES = C[:3] + ['Academic Appropriateness']
BLUE = '#0072B2'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent/'data')
    p.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent/'figures')
    a = p.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    def rows(file):
        with (a.data_dir/file).open(encoding='utf-8-sig',newline='') as f:
            return list(csv.DictReader(f))
    def save(fig, number):
        for ext in ('png','pdf'):
            fig.savefig(a.out_dir/f'Figure_{number}.{ext}', dpi=300, facecolor='white')
        plt.close(fig)
    def grid(ax,axis):
        ax.set_axisbelow(True); ax.grid(axis=axis,color='#dddddd',lw=.7)

    x = [int(r['body_word_count']) for r in rows('LICA_LEAF_100_descriptives.csv')]
    fig,ax=plt.subplots(figsize=(6.8,4.1),layout='constrained')
    ax.hist(x,bins=np.arange(175,476,25),color=BLUE,edgecolor='white')
    ax.axvline(250,ls='--',color='#9E3516',label='250-word threshold')
    ax.set(xlabel='Essay-body word count',ylabel='Number of essays'); ax.yaxis.set_major_locator(MaxNLocator(integer=True)); grid(ax,'y'); ax.legend(frameon=False)
    save(fig,1)

    A={(r['subset'],r['criterion'],r['pair']):r for r in rows('agreement.csv')}
    def forest(subset, pairs, number, limits):
        fig,ax=plt.subplots(figsize=(7.6,4.3),layout='constrained')
        for ix,(pair,marker,color) in enumerate(pairs):
            records=[A[subset,c,pair] for c in C]
            q,lo,hi=np.array([[float(r[k]) for k in ('qwk','qwk_lo','qwk_hi')] for r in records]).T
            pos=np.arange(4)+(ix-(len(pairs)-1)/2)*.22
            ax.errorbar(q,pos,xerr=[q-lo,hi-q],fmt=marker,capsize=3,color=color,label=pair.replace('-','–'))
        counts=[A[subset,c,pairs[0][0]]['n'] for c in C]
        ax.set_yticks(range(4),[f'{name} (n = {n})' for name,n in zip(NAMES,counts)])
        ax.set(xlim=limits,ylim=(3.55,-.55),xlabel='Quadratic-weighted Cohen’s kappa'); grid(ax,'x')
        if len(pairs)>1:
            ax.axvline(0,color='#777777',ls=':',lw=1)
            ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False)
        save(fig,number)
    forest('pairwise',[('Human1-Human2','o',BLUE)],2,(0,1.02))

    pairs=rows('pairs_used.csv')
    limits={r['criterion']:r for r in rows('bland_altman_descriptive.csv') if r['pair']=='Human1-Human2'}
    fig,axs=plt.subplots(2,2,figsize=(7.8,6),layout='constrained',sharex=True,sharey=True)
    for ax,c,name in zip(axs.flat,C,NAMES):
        points=Counter(((int(r['score_a'])+int(r['score_b']))/2,int(r['score_a'])-int(r['score_b'])) for r in pairs if r['criterion']==c and r['pair']=='Human1-Human2')
        coords=list(points); xx,yy=zip(*coords); ax.scatter(xx,yy,s=[9*points[k] for k in coords],color=BLUE,alpha=.72,zorder=3)
        r=limits[c]
        for key,style,col in [('bias','-','#9E3516'),('lower_loa','--','#444444'),('upper_loa','--','#444444')]:
            ax.axhline(float(r[key]),linestyle=style,color=col,lw=1)
        ax.set(title=f'{name} (n = {r["n"]})',xlim=(.75,5.25),ylim=(-1.55,1.55),xticks=range(1,6)); grid(ax,'y')
    fig.supxlabel('Mean of the two human ratings'); fig.supylabel('Human1 − Human2 (points)'); save(fig,3)

    freq={(r['criterion'],int(r['score'])):int(r['count']) for r in rows('score_frequencies.csv') if r['rater']=='Model'}
    fig,axs=plt.subplots(2,2,figsize=(7.8,5.7),layout='constrained',sharex=True,sharey=True)
    for ax,c,name in zip(axs.flat,C,NAMES):
        scores=list(range(1,6)); vals=[freq[c,k] for k in scores]; assert sum(vals)==100
        bars=ax.bar(scores,vals,color=BLUE,width=.65); ax.bar_label(bars,padding=3,fontsize=9)
        ax.set(title=name,xticks=scores,ylim=(0,85)); grid(ax,'y')
    fig.supxlabel('Model rating'); fig.supylabel('Number of essays'); save(fig,4)
    forest('matched',[('Human1-Human2','o','#222222'),('Model-Human1','s',BLUE),('Model-Human2','^','#9E3516')],5,(-.25,1.02))

if __name__=='__main__':
    main()
