import hashlib,collections
from core import *
from src.tools.inference import load_smplh
from src.tools.smplrifke_feats import smplrifkefeats_to_smpldata
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    torch.set_num_threads(2);state=torch.load(OUT/'free_best.pt',map_location='cpu',weights_only=False)
    base=ROOT/'outputs_amass/official_20260916/logs/checkpoints/last.ckpt'
    deploy=dict(adapter=state['adapter'],step=state['step'],development_score=state['score'],tasks=TASKS,ranges=RANGES,units=UNITS,frames=FRAMES,source_checkpoint=str(base),root_adapter=str(OLD/'best.pt'),source_sha256=sha(base),root_adapter_sha256=sha(OLD/'best.pt'),representation='SMPL-RIFKE205, 20Hz; numeric scalar injected during denoising')
    torch.save(deploy,OUT/'task_adapter_deploy.pt')
    records=json.loads((OUT/'data_manifest.json').read_text());training={r['family'] for r in records if r['split']=='train'};val={r['family'] for r in records if r['split']=='val'};assert not training&val
    fk=FK();smpl=load_smplh();errs=[]
    manifests=json.loads((OUT/'generated/task_adapter_manifest.json').read_text())
    for task in TASKS:
        row=next(r for r in manifests if r['task']==task and r['command_index']==4);raw=torch.from_numpy(np.load(row['path'])['motion'])
        ids=torch.tensor([0,len(raw)//2,len(raw)-1]);bd=smplrifkefeats_to_smpldata(raw)
        with torch.no_grad():p=smpl(bd['poses'][ids],bd['trans'][ids],jointstype='smpljoints');got=fk(raw[None],canonical=False)[0,ids]
        errs.append(dict(task=task,max_joint_error_m=float((p-got).abs().max())))
    assert max(r['max_joint_error_m'] for r in errs)<.001,errs
    audit=dict(counts={s:{t:sum(r['split']==s and r['task']==t for r in records) for t in TASKS} for s in ['train','val']},training_unique_families=len(training),validation_unique_families=len(val),family_overlap=0,fk_parity=errs,human_height_m=fk.height,train_caption_selection='automatic task keywords, source family deduplication; not manual semantic certification',free_generation_training='4 prompts/task also used in confirmation; new random noise seeds are held out',selected_step=state['step'],development_score=state['score'],stage1='3300 supervised steps did not improve held-out denoising loss; best remained initialization',stage2='8800 free-generation finetuning steps; selected by development-noise free-generation metric',note='No original or root-control weights were changed. No simulator used in adapter training.')
    save('data_audit.json',audit)
    files=list((OUT/'code').glob('*.py'))+[base,OLD/'best.pt',OUT/'task_adapter_deploy.pt',OUT/'skeleton.npz',OUT/'official_human_skeleton.npz',ROOT/'work/g1_sonic_official/gear_sonic_deploy/policy/release/model_encoder.onnx',ROOT/'work/g1_sonic_official/gear_sonic_deploy/policy/release/model_decoder.onnx']
    save('delivery/provenance.json',dict(files={str(p):sha(p) for p in files},controller_repo_commit='b042411fae38ee4d1af9aac82a37a1f8d14d6dd0',mdm_metric_source_repo_commit='d5c44fc864d3b8fc784e227888043ddfe4c0fde5'))
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
