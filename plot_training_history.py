from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
runs=[('Near-OOD',root/'results/outputs/ood_v3_qwen3_4b_gate_lora/checkpoint-1491/trainer_state.json'),('Full-data retraining',root/'results/outputs/full_qwen3_4b_gate_lora/checkpoint-1815/trainer_state.json')]
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
colors={'loss':'#2563eb','lm_loss':'#0d9488','gate_loss':'#f59e0b','eval_loss':'#dc2626'}
count=0
for ax,(label,path) in zip(axes,runs):
 logs=json.loads(path.read_text())['log_history']
 for field,color in colors.items():
  points=[(x['step'],x[field]) for x in logs if 'step' in x and isinstance(x.get(field),(int,float))]
  if points:
   x,y=zip(*points);ax.plot(x,y,label=field,color=color,marker='o' if field=='eval_loss' else None,markersize=4,linewidth=1.2,alpha=.9);count+=len(points)
 ax.set(title=label,xlabel='Optimization step',ylabel='Recorded loss');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
fig.suptitle('Recorded GPU training history (existing logs; no rerun)',fontsize=12)
out=root/'figures/recorded_training_loss.png';out.parent.mkdir(exist_ok=True);fig.savefig(out,dpi=160,facecolor='white')
print(json.dumps({'plotted_recorded_loss_values':count,'source_logs':[str(p.relative_to(root)) for _,p in runs]}))
