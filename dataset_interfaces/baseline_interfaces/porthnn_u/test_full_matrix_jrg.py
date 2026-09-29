# Author: Simone Betteti
"""Structural checks for the three-link full-matrix PortHNN-u."""
import sys
from pathlib import Path
import jax, jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).parent))
from porthnn_u_full_matrix_jax import initialize, hamiltonian, matrices, vector_field, rk4_step

def main():
 p=initialize(); x=jax.random.normal(jax.random.PRNGKey(1),(4,6)); u=jnp.ones((4,1))*.2
 h=jax.vmap(lambda z:hamiltonian(p,z))(x); grad=jax.vmap(lambda z:jax.grad(hamiltonian,1)(p,z))(x)
 j,r,g=jax.vmap(lambda z:matrices(p,z))(x); vf=jax.vmap(lambda z,v:vector_field(p,z,v))(x,u)
 assert h.shape==(4,) and grad.shape==(4,6) and j.shape==r.shape==(4,6,6) and g.shape==(4,6,1)
 assert float(jnp.max(jnp.abs(j+j.swapaxes(-1,-2)))) < 1e-10
 assert float(jnp.max(jnp.abs(r-r.swapaxes(-1,-2)))) < 1e-10 and float(jnp.min(jnp.linalg.eigvalsh(r))) >= -1e-10
 skew=jnp.einsum('bi,bij,bj->b',grad,j,grad); rhs=-jnp.einsum('bi,bij,bj->b',grad,r,grad)+jnp.einsum('bi,bij,bj->b',grad,g,u)
 assert float(jnp.max(jnp.abs(skew))) < 1e-9
 assert float(jnp.max(jnp.abs(jnp.einsum('bi,bi->b',grad,vf)-rhs))) < 1e-9
 # Each branch must receive a gradient; G only does so for nonzero input.
 loss=lambda q:jnp.sum(jax.vmap(lambda z,v:vector_field(q,z,v))(x,u)**2)
 grads=jax.grad(loss)(p)
 for name in ('hamiltonian','j_head','r_head','g_head'): assert any(bool(jnp.any(jnp.abs(a)>0)) for a in jax.tree_util.tree_leaves(grads[name]))
 assert bool(jnp.all(jnp.isfinite(jax.vmap(lambda z,v:rk4_step(p,z,v,.05))(x,u))))
 print('full-matrix PortHNN-u structural tests passed')
if __name__=='__main__': main()
