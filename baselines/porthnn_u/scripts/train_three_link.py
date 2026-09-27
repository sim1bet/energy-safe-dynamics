"""Reproducible rollout trainer for the n=3 full-matrix PortHNN-u."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np, optax
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from porthnn_u.three_link import initialize, parameter_counts, rollout

ROOT=Path(__file__).resolve().parents[3]
DATA=Path(os.environ.get('PORTHNN_U_THREE_LINK_DATA', 'datasets/deep_dissipative_nlink_n3/parameters'))
DT=.05
def write(path, value):
 path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def mass(q):
 # NLinkPendulum defaults: absolute angles, l_i=1/3, m_i=1/6.
 idx=jnp.arange(3); i=idx[:,None]; k=idx[None,:]
 coeff=(3-jnp.maximum(i,k))/54.
 return coeff*jnp.cos(q[:,None]-q[None,:])
def canonical(states):
 return np.asarray(jax.vmap(lambda s:jnp.concatenate((s[:3],mass(s[:3])@s[3:])))(jnp.asarray(states.reshape(-1,6)))).reshape(states.shape)
def windows(states, inputs, horizon):
 # Entire trajectories remain intact; each output window is within one trajectory.
 x=[]; u=[]; y=[]
 for tr,command in zip(states,inputs):
  for start in range(len(tr)-horizon): x.append(tr[start]); u.append(command[start:start+horizon]); y.append(tr[start+1:start+horizon+1])
 return np.asarray(x),np.asarray(u),np.asarray(y)
def tree_l2(tree): return sum(jnp.sum(a*a) for a in jax.tree_util.tree_leaves(tree) if hasattr(a,'dtype'))
def main():
 global DATA
 ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--data-dir',type=Path,default=DATA,help='directory containing ex1/ex2 nlink .npy files'); ap.add_argument('--updates',type=int,default=20000); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
 DATA=a.data_dir
 train_raw=np.load(DATA/'ex1_n3nlink.train.state.npy'); val_raw=np.load(DATA/'ex2_n3nlink.train.state.npy')
 train_u=np.load(DATA/'ex1_n3nlink.train.input.npy'); val_u=np.load(DATA/'ex2_n3nlink.train.input.npy')
 train=np.asarray(canonical(train_raw), dtype=np.float64)
 val=np.asarray(canonical(val_raw), dtype=np.float64)
 train_u=np.asarray(train_u, dtype=np.float64); val_u=np.asarray(val_u, dtype=np.float64)
 mean=train.reshape(-1,6).mean(0); std=train.reshape(-1,6).std(0).clip(1e-6)
 # Preserve physical canonical coordinates inside the pH model.  Train-only
 # channel scales are used only to weight prediction error.
 um=train_u.reshape(-1,1).mean(0); us=train_u.reshape(-1,1).std(0).clip(1e-6)
 horizon=16; tx,tu,ty=windows(train,train_u,horizon); vx,vu,vy=windows(val,val_u,horizon)
 if a.smoke: a.updates=2
 if not len(tx): raise RuntimeError('dataset has no valid 16-step trajectory windows')
 p=initialize(0,input_dim=train_u.shape[-1]); schedule=optax.warmup_cosine_decay_schedule(0.,3e-4,500,max(a.updates,501),3e-6)
 opt=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(schedule,weight_decay=1e-6)); state=opt.init(p)
 weights=jnp.linspace(1.,2.,horizon); weights/=weights.mean()
 def loss(q,x,u,target):
  pred=jax.vmap(lambda z,v:rollout(q,z,v,DT)[1:])(x,u)
  data=jnp.mean(weights[None,:,None]*((pred-target)/jnp.asarray(std))**2)
  reg=1e-6*(tree_l2(q['hamiltonian'])+tree_l2(q['j_head'])+tree_l2(q['r_head'])+tree_l2(q['g_head']))
  return data+reg
 @jax.jit
 def update(q,s,x,u,y):
  value,grad=jax.value_and_grad(loss)(q,x,u,y); upd,s=opt.update(grad,s,q); return optax.apply_updates(q,upd),s,value
 validate=jax.jit(loss); rng=np.random.default_rng(0); best=float('inf'); bestp=p; hist=[]; started=time.time()
 for step in range(a.updates):
  ind=rng.integers(len(tx),size=min(32,len(tx))); p,state,value=update(p,state,jnp.asarray(tx[ind]),jnp.asarray(tu[ind]),jnp.asarray(ty[ind]))
  if step==0 or (step+1)%200==0 or step+1==a.updates:
   v=float(validate(p,jnp.asarray(vx),jnp.asarray(vu),jnp.asarray(vy))); hist.append({'update':step+1,'train_loss':float(value),'validation_loss':v})
   if np.isfinite(v) and v<best: best=v; bestp=jax.tree_util.tree_map(lambda z:z.copy(),p)
   print(hist[-1],flush=True)
 out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(out/'checkpoint_best.npz',**{f'{k}_{i}_{n}':np.asarray(v) for k in ('hamiltonian','j_head','r_head','g_head') for i,l in enumerate(bestp[k]) for n,v in l.items() if v is not None})
 np.savez_compressed(out/'normalization.npz',state_mean=mean,state_std=std,input_mean=um,input_std=us)
 write(out/'training_history.json',{'history':hist}); write(out/'resolved_config.json',{'state_dim':6,'input_dim':int(train_u.shape[-1]),'dt':DT,'horizons':[16,32,64],'effective_horizon':16,'reason':'initial development run uses the shortest declared horizon; longer-horizon curriculum is deferred until this baseline is finite','updates':a.updates,'parameter_counts':parameter_counts(bestp),'best_validation_rollout_loss':best,'elapsed_seconds':time.time()-started,'precision':'float64'})
 files=['ex1_n3nlink.train.state.npy','ex1_n3nlink.train.input.npy','ex2_n3nlink.train.state.npy','ex2_n3nlink.train.input.npy','ex1_n3nlink.test.state.npy','ex1_n3nlink.test.input.npy']
 write(out/'provenance.json',{'dataset_files_sha256':{f:sha(DATA/f) for f in files},'splits':{'train':'ex1 train trajectory 0','validation':'ex2 train trajectory 0','test':'ex1 test trajectories 0..9'},'state_source':'dataset (q, qdot); transformed to p=M(q)qdot using NLinkPendulum default mass matrix','input':'one torque channel, applied to first generalized coordinate by source simulator','python':platform.python_version(),'jax':jax.__version__,'source_tree_sha256':hashlib.sha256((ROOT/'porthnn_u/three_link.py').read_bytes()).hexdigest()})
if __name__=='__main__': main()
