# Modular $(H,J,R,G)$ Migration

The legacy PortHNN-u vector field remains unchanged:

$$
\dot z = [\nabla_p H, -\nabla_q H + N \odot \nabla_p H + F(u)].
$$

For even latent dimension $d=2n$, compatibility maps this to the canonical $J_0$ and momentum-block $R_{legacy}=\operatorname{diag}(0,-N)$. The generalized implementation in `modular.py` uses independently parameterized state-only storage $H(z)$, packed-skew $J(z)$, Cholesky-PSD $R(z)$, and full state-dependent $G(z)u$.

Legacy checkpoints are only valid with `legacy_porthnn_u: true`. Generalized checkpoints have model schema version 2 and must not be loaded as legacy parameters.