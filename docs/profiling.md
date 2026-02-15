# GPAW ETDM-FDPW GPU Profiling Notes

Date: 2026-02-15  
Source logs:
- `log/gpaw_etdm_gpu_trace.txt`
- `log/gpaw_trace_gpu.json`

## Run context

- k-point mesh is confirmed to be `[4, 4, 4]`:
  - `config.yaml:53`
  - `log/gpaw_etdm_gpu_trace.txt:79`
- Calculation mode: direct optimization (`etdm-fdpw`), not SCF.

## Does `P_ani` get recomputed many times within one ETDM step?

Short answer: it is **not** repeatedly recomputed in the same phase if wavefunctions do not change.  
It is recomputed after explicit cache invalidation, which ETDM performs after wavefunction updates.

### Relevant code path

1. ETDM updates wavefunctions:
   - `gpaw/new/pwfd/etdm.py:102`
2. ETDM explicitly invalidates projector cache:
   - `gpaw/new/pwfd/etdm.py:105` (`wfs._P_ani = None`)
3. `P_ani` is lazily rebuilt on first access:
   - `gpaw/new/pwfd/wave_functions.py:125-131`
   - actual build call: `self.pt_aiX.integrate(self.psit_nX, self._P_ani)`
4. Subsequent accesses in that stage reuse cached `self._P_ani` until next invalidation.

### Where `P_ani` is consumed after rebuild

- Nonlocal Hamiltonian application:
  - `gpaw/new/pwfd/etdm.py:196-199`
- Gradient projection with overlap correction:
  - `gpaw/new/pwfd/etdm.py:214-216`
- Density matrix update path:
  - `gpaw/new/density.py:220-223`
  - `gpaw/new/pwfd/wave_functions.py:149-154`
  - `gpaw/new/wave_functions.py:148-157`

So the expensive part is not "useless repeated recompute at every call site", but "recompute once per invalidation phase", and ETDM invalidates often by design.

## Math to code mapping

Projector coefficients:

$$
F_{GI} = \langle G \mid \tilde p_I \rangle,\quad
P_{nI} = \langle \tilde p_I \mid \tilde \psi_n \rangle
= \sum_G \tilde \psi_{nG} F_{GI}^{*}
$$

Nonlocal PAW term:

$$
c_{nI} = \sum_J P_{nJ} D_{JI},\quad
(\hat V_{NL}\tilde \psi_n)_G = \frac{1}{dv}\sum_I c_{nI} F_{GI}
$$

Atomic density matrix contribution (collinear case):

$$
D^{(a)}_{ij} \mathrel{+}= \sum_n f_n\, P_{ni}^{(a)*} P_{nj}^{(a)}
$$

This corresponds to GEMM-heavy kernels in `PWLFC.integrate()` and `PWLFC.add()`.

## Aggregated hotspot report (CPU+GPU self-time combined)

From `log/gpaw_trace_gpu.json`:

| Module | Time (s) | Share |
|---|---:|---:|
| PWLFC/projector | 5.834 | 32.37% |
| Density update | 3.223 | 17.88% |
| GEMM/BLAS | 2.677 | 14.85% |
| FFT | 1.756 | 9.74% |
| Precondition/kinetic | 1.704 | 9.46% |
| Other | 1.300 | 7.21% |
| Local potential | 0.830 | 4.60% |
| Potential build/XC | 0.700 | 3.88% |

## Interpretation

- If "wave coeff -> density matrix" means only `D += P^* f P`, then it is **not** the single largest item.
- If it means the full projector contraction chain (`psi_G -> P_ani`, nonlocal projector add/integrate, related GEMMs), then yes: this family is the dominant cost center.
- `pwlfc_expand_gpu` is not rerunning radial SBT every iteration.  
  Radial transform is initialized once in `PWLFC.initialize()` (`gpaw/core/pwacf.py:172`, guarded by `:144`), while per-iteration work is phase/spherical-harmonic expansion and GEMM contractions.

## Jrystal PAW (same config) timing snapshot

Run command:

```bash
/home/aiops/zhaojx/venv/aisci/bin/jrystal -m energy -c config.yaml
```

Run notes:
- Output was redirected to `log/jrystal_energy_profile.raw.log` to avoid terminal I/O overhead.
- The run was stopped after collecting stable timing behavior (up to ~303 iterations).

### Measured stage times (from log)

| Stage | Time (s) |
|---|---:|
| Local pseudopotential init | 1.81 |
| SBT init | 5.17 |
| Nonlocal potential init | 15.97 |
| Nonlocal deploy | 0.00 |

Precompute total: `22.95 s`

### Optimization loop behavior

- Progress reached `303/10000` at elapsed `04:10` (inside tqdm loop timer).
- This implies loop elapsed of about `250 s` for first 303 iterations.
- Early iterations are warm-up heavy (JIT/compile/autotune effect), then throughput stabilizes around:
  - `2.7 ~ 3.0 it/s`
  - i.e. `~0.34 ~ 0.37 s/it` steady-state

### What is the main time consumer?

For the observed window (init + first 303 iterations):

| Category | Time (s) | Share |
|---|---:|---:|
| Optimization loop | 250.00 | 91.59% |
| Nonlocal init | 15.97 | 5.85% |
| SBT init | 5.17 | 1.89% |
| Local init | 1.81 | 0.66% |

Conclusion:
- Overall dominant cost is the optimization loop (`update` / free-energy+grad path), not pseudopotential precompute.
- Within precompute, nonlocal initialization is the largest part.

## Jrystal JAX trace: 8-GPU vs 1-GPU (`k_grid_sizes=[2,2,2]`)

Run setup:
- 8-GPU k-parallel:
  - `parallel_over_k_mesh=True`
  - trace: `log/jax_trace_update_after_factor_k222_i20/plugins/profile/2026_02_15_14_40_34/jiaxi-9233fc-job-ghv5f.trace.json.gz`
- 1-GPU (no k-parallel):
  - `parallel_over_k_mesh=False`
  - trace: `log/jax_trace_update_after_factor_k222_single_i20/plugins/profile/2026_02_15_14_48_55/jiaxi-9233fc-job-ghv5f.trace.json.gz`

Both traces are 1-step steady-state captures (`start_iter=20`, `steps=1`).

### Kernel bucket comparison (inside `jrystal_update`)

| Case | `jrystal_update` | Kernel total | Collective | QR/EIGH related | GEMM | FFT | Memcpy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8-GPU (`k`-parallel) | 59.761 ms | 33.885 ms | 14.398 ms (42.49%) | 3.879 ms (11.45%) | 6.486 ms (19.14%) | 3.275 ms (9.67%) | 5.531 ms (16.32%) |
| 1-GPU (no `k`-parallel) | 25.585 ms | 11.176 ms | 0.000 ms (0.00%) | 3.464 ms (30.99%) | 4.424 ms (39.59%) | 0.660 ms (5.91%) | 2.502 ms (22.38%) |

Key observation:
- In 8-GPU mode, collectives (`nccl AllGather/AllReduce/ReduceScatter`) are the largest kernel bucket.
- In 1-GPU mode, collectives disappear, and bottleneck shifts to local linear algebra (QR/GEMM) plus memcpy.

### Function-level attribution (`k=[2,2,2]`, 1-GPU)

Using trace event field `args.name` (TensorBoard scope stack), the heavy kernels map to:

1. QR path from plane-wave coefficient orthogonalization:
   - Scope: `jit(update)/jit(main)/jvp(jit(_qr))/jit(qr)/geqrf`
   - Dominant kernel: `geqr2_gmem_domino`
   - Time: `~3.05 ms` in this step
   - Code path:
     - `jrystal/calc/calc_ground_state_energy_paw.py:280`
     - `jrystal/_src/pw.py:136`
     - `jrystal/_src/unitary_module.py:75`

2. GEMM from projector overlap contraction:
   - Scope: `jit(update)/jit(main)/jvp(abcdef,bghdef->abcgh)/dot_general`
   - Time: `~1.94 ms` in this step
   - This matches:
     - `jrystal/calc/calc_ground_state_energy_paw.py:220-223`
     - `_f_matrix = einsum(coeff.conj(), proj_pw_overlap, "s k band x y z, k beta phi x y z -> s k band beta phi")`

3. Smaller GEMMs from PAW density-matrix contraction:
   - Scopes:
     - `jit(update)/jit(main)/jvp(abcd,abc,abce->de)/dot_general`
     - `jit(update)/jit(main)/transpose(jvp(abcd,abc,abce->de))/dot_general`
   - This matches:
     - `jrystal/calc/calc_ground_state_energy_paw.py:244-249`
     - `einsum(_f_matrix.conj(), occ, _f_matrix, "s k band proj1, s k band, s k band proj2 -> proj1 proj2")`

4. Additional GEMM from QR backward internals:
   - Scope: `jit(update)/jit(main)/jvp(jit(_qr))/jit(qr)/householder_product`
   - Time: `~0.54 ms`
   - This is still part of QR autodiff, not a separate physics operator.

Note:
- No EIGH kernels were found inside the traced `jrystal_update` step.
- This is consistent with moving ultrasoft `qr/eigh` to precompute stage in `get_ultrasoft_coeff_fun` (`jrystal/pseudopotential/ultrasoft.py:121-124`), outside `update`.

### Why there is still communication with k-point sharding

Even when data is sharded by `k`, several operations reduce over the global `k` axis, which requires collectives:

1. Density reduction over `k` and `band`:
   - `jrystal/_src/pw.py:278`
   - `dens = jnp.einsum('skb...,skb->s...', dens, occupation)`
2. Kinetic energy reduction over `k` and `band`:
   - `jrystal/_src/energy.py:178`
   - `e_kin = jnp.sum(e_kin * occupation) / 2`
3. PAW atomic density matrix accumulation over `s,k,band`:
   - `jrystal/calc/calc_ground_state_energy_paw.py:244-249`
4. Occupation simplex projector uses global flatten/projection (`num_kpts * num_bands`):
   - `jrystal/_src/occupation.py:353-363`
   - and global sums/sorts in `proj(...)` (`jrystal/_src/occupation.py:313-340`)
5. Scalar loss/grad path in `update` requires cross-shard reduction:
   - `jrystal/calc/calc_ground_state_energy_paw.py:402-410`

So "different GPUs own different k-points" is true for storage, but global reductions in physics/optimization still force communication.

## 2GB profiler limit explanation

The error
`tensorflow.profiler.XSpace exceeded maximum protobuf size of 2GB`
means the profiler's serialized trace object (`XSpace` protobuf) exceeded protobuf's per-message size limit during export.

Important:
- This is a trace export size limit, not GPU memory capacity.
- In this run the attempted serialized size was above 2GB (e.g. `3385529225` bytes in `log/jax_trace_update_after_factor_capture.log`).
- When this happens, `stop_trace()` fails and trace artifacts become incomplete.

Practical mitigation:
- Trace fewer steps (`JRYSTAL_JAX_TRACE_STEPS=1`)
- Trace later steady-state iterations (`start_iter` after compile/warmup)
- Reduce device count / sharding complexity for profiling runs
- Avoid capturing compile-heavy windows
