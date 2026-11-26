"""Batch-one end-to-end latency distribution for CPU, CUDA, and RTU."""
from __future__ import annotations
import json, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
from rtbeam.decoders import PRESETS
from rtbeam.golden import ShotSet
from rtbeam.gpu.beam import FLOAT32, GpuBeamSearch
from rtbeam.ionq import IonQBeamSearch
from rtbeam.problem import DecodingProblem
ROOT=Path(__file__).resolve().parents[1]
CASE='bb144_z_p0.001'
PRESET='beam8_230'
CIRCUITS=ROOT/'external'/'BeamSearchDecoder'/'StimCircuit'

def main(backend: str) -> None:
    shotset=ShotSet.load(ROOT/'data'/'golden'/CASE/'shots.npz')
    shots=shotset.syndromes
    problem=DecodingProblem.from_stim_file(CIRCUITS/f'{shotset.stem}.stim')
    t=time.perf_counter()
    if backend=='cpu': decoder=IonQBeamSearch(problem,PRESETS[PRESET])
    elif backend=='cuda': decoder=GpuBeamSearch(problem,PRESETS[PRESET],FLOAT32,backend='cuda')
    elif backend=='rt': decoder=GpuBeamSearch(problem,PRESETS[PRESET],FLOAT32,backend='rt')
    elif backend=='rt-cache': decoder=GpuBeamSearch(problem,PRESETS[PRESET],FLOAT32,backend='rt',rt_cache_neighbors=True)
    else: raise ValueError(backend)
    setup_s=time.perf_counter()-t
    decoder.decode_batch(shots[:min(1024,len(shots))])
    seconds=np.empty(len(shots),dtype=np.float64)
    for i,syndrome in enumerate(shots):
        start=time.perf_counter(); decoder.decode_batch(syndrome[None,:]); seconds[i]=time.perf_counter()-start
        if (i+1)%2000==0: print(f'{backend}: {i+1}/{len(shots)}',flush=True)
    report={'backend':backend,'case':CASE,'preset':PRESET,'host':host_info(),'decoder_setup_seconds':setup_s,
            'latency_shots':len(seconds),'latency_ms':{'mean':float(np.mean(seconds)*1e3),
            'p50':float(np.percentile(seconds,50)*1e3),'p99_9':float(np.percentile(seconds,99.9)*1e3),
            'max':float(np.max(seconds)*1e3)}}
    out=ROOT/'results'/f'latency_{backend}.json'; out.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2),flush=True)

def host_info() -> dict[str,str]:
    smi=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,clocks.max.sm','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    return {'gpu':smi,'cpu':platform.processor(),'loadavg':' '.join(f'{x:.1f}' for x in os.getloadavg()),'cpus':str(os.cpu_count())}
if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('usage: bench_latency_batch1.py cpu|cuda|rt|rt-cache')
    main(sys.argv[1])
