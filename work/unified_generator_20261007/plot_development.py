import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D=Path('/home/pku/frankenmotion/outputs_amass/unified_generator_20261007');out=D/'report';out.mkdir(exist_ok=True)
names=['teacher_development','shared_v1_11000_development','shared_v2_33000_development','shared_v3_4400_development','shared_v4_4400_development','shared_v4_11000_development'];labels=['Old routed','v1 narrow','v2 preserve','v3 wider','v4 selected','v4 later'];rows=[]
for n in names:
 h=json.loads((D/'generation'/n/'audit.json').read_text());g=json.loads((D/'general_evaluation'/n/'eleven_audit.json').read_text())['aggregate'];rows.append(dict(name=n,human=h,physics=g))
x=np.arange(len(names));fig,ax=plt.subplots(figsize=(10,4));ax.bar(x-.18,[r['human']['joint_pass'] for r in rows],.36,label='Human reference',color='#258abe');ax.bar(x+.18,[r['physics']['joint'] for r in rows],.36,label='Same frozen tracker',color='#e08b34');ax.set_xticks(x);ax.set_xticklabels(labels);ax.set_ylim(0,110);ax.set_ylabel('Command + event passes /110');ax.set_title('Development search only; not an independent test or controlled ablation');ax.legend();ax.grid(axis='y',alpha=.15)
for i,r in enumerate(rows):
 for d,v in [(-.18,r['human']['joint_pass']),(.18,r['physics']['joint'])]:ax.text(i+d,v+1,str(v),ha='center',fontsize=9)
fig.tight_layout();fig.savefig(out/'development_search.png',dpi=180);plt.close(fig);(out/'development_search.json').write_text(json.dumps(rows,indent=2))
