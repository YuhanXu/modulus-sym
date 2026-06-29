#!/usr/bin/env python3
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE='/work/PaddleX_QA_test_new_science_max_step/limerock-limerock_hFTB-limerock_thermal'
OUT=BASE+'_step0_20000_loss_compare.png'
pat=re.compile(r'\[step:\s*(\d+)\].*?loss:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)')

def read(p):
    d={}
    with open(p,errors='ignore') as f:
        for line in f:
            m=pat.search(line)
            if m:
                s=int(m.group(1))
                if s<=20000:
                    d[s]=float(m.group(2))
    return d

dy=read(BASE+'_dy.log')
ci=read(BASE+'_cinn.log')
common=sorted(set(dy)&set(ci))
steps=np.array(common)
dy_loss=np.array([dy[s] for s in common])
ci_loss=np.array([ci[s] for s in common])
delta=ci_loss-dy_loss
out=np.abs(delta)>1e-2

fig,axes=plt.subplots(3,1,figsize=(14,12),sharex=True)
axes[0].plot(steps,dy_loss,label='dy',lw=0.8)
axes[0].plot(steps,ci_loss,label='CINN',lw=0.8,ls='--')
axes[0].set_ylabel('Loss')
axes[0].set_title('limerock/limerock_hFTB/limerock_thermal loss comparison (step 0-20000)')
axes[0].grid(True,alpha=0.3); axes[0].legend()
axes[1].semilogy(steps,np.maximum(np.abs(dy_loss),1e-30),label='|dy loss|',lw=0.8)
axes[1].semilogy(steps,np.maximum(np.abs(ci_loss),1e-30),label='|CINN loss|',lw=0.8,ls='--')
axes[1].set_ylabel('|Loss| log scale')
axes[1].grid(True,alpha=0.3); axes[1].legend()
axes[2].plot(steps,delta,label='CINN - dy',lw=0.8)
axes[2].axhline(0,color='black',lw=0.5)
axes[2].axhline(1e-2,color='red',ls=':',lw=0.8)
axes[2].axhline(-1e-2,color='red',ls=':',lw=0.8)
axes[2].scatter(steps[out],delta[out],s=5,c='red',label='|delta|>1e-2')
axes[2].set_xlabel('Step'); axes[2].set_ylabel('Delta loss')
axes[2].set_title(f'Delta loss, outliers: {int(out.sum())}/{len(delta)}')
axes[2].grid(True,alpha=0.3); axes[2].legend()
plt.tight_layout(); plt.savefig(OUT,dpi=120); plt.close()
print(OUT)
