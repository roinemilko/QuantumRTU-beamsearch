"""Three two-panel latency/throughput figures using existing percentile and batch data."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
CASE='bb144_z_p0.001'
PRESETS=('beam8_230','beam32_340','beam64_640')
BEAM_LABELS=('beam8','beam32','beam64')
COLORS={'cpu':'#2a78d6','cuda':'#eb6834','rt':'#1baf7a','rt-cache':'#8b5fbf'}
LABELS={'cpu':'CPU (single thread)','cuda':'Standard CUDA','rt':'RTU','rt-cache':'RTU (cached edge IDs)'}
SHORT={'cpu':'CPU','cuda':'CUDA','rt':'RTU','rt-cache':'RTU cache'}
PLOTS=[
 ('throughput_cpu_cuda.png',('cpu','cuda'),'CPU vs standard CUDA'),
 ('throughput_cpu_cuda_rtu.png',('cpu','cuda','rt'),'CPU vs standard CUDA vs RTU'),
 ('throughput_cuda_rtu_optimized.png',('cuda','rt','rt-cache'),'CUDA vs RTU vs optimized RTU'),
]
# Paper reports 42.5 us/round for R=100, batch=256, A100. 12-round equivalent shown only as a reference.
SOTA_MS=.510

def load_json(name): return json.loads((RESULTS/name).read_text())

def main() -> None:
    throughput={k:load_json(f'throughput_{k}.json') for k in ('cpu','cuda','rt','rt-cache')}
    cpu=load_json('cpu_baseline.json')
    full={'cuda':load_json('gpu_baseline.json'),'rt':load_json('rt_baseline.json'),
          'rt-cache':load_json('rt-cache_baseline.json')}
    latency={'cpu':cpu}
    for backend,report in full.items(): latency[backend]=report
    for filename,backends,title in PLOTS:
        fig,(left,right)=plt.subplots(1,2,figsize=(13,6),facecolor='white',gridspec_kw={'width_ratios':(1,1.25)})
        latency_panel(left,backends,latency)
        throughput_panel(right,backends,throughput)
        fig.suptitle(title,x=.01,y=.985,ha='left',fontsize=15)
        gpu=throughput['cuda']['host']['gpu']
        fig.text(.01,.012,
          f"BB[[144,12,12]] · memory-Z · p=0.001 · float32 · GPU: {gpu}\n"
          "Latency uses batch-one means and p99.9 across beam widths 8/32/64. Throughput uses beam8 and batches through 100,000. "
          "SOTA reference normalizes 42.5 us/round at R=100 (batch256, A100) to 12 rounds; not the same workload.",
          fontsize=7.5,color='#555')
        fig.tight_layout(rect=(0,.06,1,.94))
        out=RESULTS/filename
        fig.savefig(out,dpi=180,bbox_inches='tight')
        plt.close(fig)
        print(out)

def latency_values(name,latency,preset):
    if name=='cpu': return latency['cpu'][CASE][preset]
    entry=latency[name][f'{CASE}/{preset}/float32']
    return entry['latency']

def latency_panel(ax,backends,latency) -> None:
    n=len(backends)
    width=.72/n
    offsets=[(j-(n-1)/2)*width for j in range(n)]
    for i,preset in enumerate(PRESETS):
        for j,name in enumerate(backends):
            values=latency_values(name,latency,preset)
            mean=values['mean_ms']
            p999=values['p999_ms']
            x=i+offsets[j]
            ax.bar(x,mean,width=width*.9,color=COLORS[name],zorder=2)
            ax.scatter(x,p999,marker='D',s=32,color=COLORS[name],edgecolor='white',linewidth=1,zorder=3)
            ax.text(x,mean*1.10,f'{mean:.2f}',ha='center',va='bottom',fontsize=6.8,rotation=0)
            ax.text(x+width*.22,p999,f'{p999:.1f}',ha='left',va='center',fontsize=6.5)
    ax.axhline(SOTA_MS,color='#222',lw=1.2,ls=':',zorder=1)
    ax.set_yscale('log')
    ax.set_ylim(.25,100)
    ax.set_xticks(range(len(PRESETS)),BEAM_LABELS)
    ax.set_ylabel('Latency per 12-round decode (ms; batch = 1)')
    ax.set_title('Latency: bars = mean, ◆ = p99.9',loc='left',fontsize=11)
    handles=[Patch(color=COLORS[k],label=LABELS[k]) for k in backends]
    handles += [Line2D([],[],marker='D',ls='',color='#555',label='p99.9'),
                Line2D([],[],color='#222',ls=':',label='SOTA normalized ref: 0.510 ms / 12 rounds')]
    ax.legend(handles=handles,frameon=False,fontsize=7.2,loc='upper left')
    style(ax)

def throughput_panel(ax,backends,throughput) -> None:
    for name in backends:
        values=throughput[name]['batch_throughput_shots_per_s']
        batches=sorted(int(k) for k in values)
        rates=[values[str(b)] for b in batches]
        legend= 'CPU throughput (Aalto host, 1 thread)' if name=='cpu' else LABELS[name]
        ax.plot(batches,rates,marker='o',ms=4.5,lw=2,color=COLORS[name],label=legend)
        if name!='cpu':
            peak=max(range(len(rates)),key=rates.__getitem__)
            suffix=' (best tested)' if peak==len(rates)-1 else ' (measured peak)'
            offset=(6,9 if name!='rt' else -16)
            ax.annotate(f'{batches[peak]:,}{suffix}',(batches[peak],rates[peak]),xytext=offset,
                        textcoords='offset points',color=COLORS[name],fontsize=7.2,ha='left')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Shots per decode_batch call')
    ax.set_ylabel('Throughput (12-round shots/s)')
    ax.set_title('Throughput vs batch size (beam8)',loc='left',fontsize=11)
    ax.legend(frameon=False,fontsize=7.6,loc='upper left')
    style(ax)

def style(ax) -> None:
    ax.grid(True,which='major',axis='y',color='#dedede',lw=.8,zorder=0)
    ax.grid(True,which='minor',axis='y',color='#eeeeee',lw=.5,zorder=0)
    for side in ('top','right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors='#555',length=0)

if __name__=='__main__': main()
