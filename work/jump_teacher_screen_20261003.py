"""Real training-set mocap as a positive control, not a FrankenMotion result."""
import sys,json
sys.path.insert(0,'/home/pku/frankenmotion/work')
import gmr_probe_20261003 as g
ROOT=g.OUT;g.OUT=ROOT/'jump_teacher_screen';g.OUT.mkdir(exist_ok=True);g.tr.rt.EFF[[4,5,10,11,13,14]]=50;g.tr.rt.ARM[[4,5,10,11,13,14]]=2*.003609725
audit=json.loads((ROOT/'jump_training_exemplars/audit.json').read_text());selected=[r for r in audit if r['metrics']['event_pass'] and r['metrics']['quantity']>=.18]
if __name__=='__main__':
 g.init();rows=[]
 for r in selected:
  row=dict(task='jump',source='mocap_'+r['key'],seed=None,command=r['metrics']['quantity'],command_index=0,path=str(ROOT/'jump_training_exemplars'/(r['key']+'.npz')),caption=r['caption'],family=r['family'],source_type='real train-set mocap, selected by event pass and lowest root ballistic residual; NOT generated')
  row=g.run((row,'uniform'));rows.append(row);print(row,flush=True)
 (g.OUT/'results.json').write_text(json.dumps(rows,indent=2,default=lambda x:x.item()))
