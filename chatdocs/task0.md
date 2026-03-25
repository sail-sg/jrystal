# 核心架构怎么设计



## 我的需求：

- 保留 `_src` 作为当前 kernel base
- 在其上重新建立稳定的workflow 层。
- 重新设计用户输入接口，如何直接根据用户输入设置执行相应的计算。
- 不希望有太多封装，造成难以维护，层次要干净，整洁，清楚，易懂。

核心对象

- `Crystal`
- `KSampling`
- `PlaneWaveBasis`
- `SolverSpec`
- `ExecutionPlan`
- `RuntimeContext`
- `GroundStateResult`
- `BandStructureResult`

以及其他的必要的对象，（帮我设计，想想怎样设计这些抽象比较好）


### `KSampling` 该怎么设计

我建议不要把 `KPointMesh` 和 `KPointPath` 完全拆成两套顶层对象。

更好的方案是统一成一个：

- `KSampling`

它有两种 mode：

- `mode="mesh"`
- `mode="path"`

统一字段可以有：

- `kpts`
- `weights`
- `labels`
- `segments`
- `source`

其中：

- mesh 模式重点使用 `weights`
- path 模式重点使用 `labels/segments`

这样做的好处是：

- workflow 接口统一
- band / ground-state 都能消费同一类对象
- 后面加 symmetry reduction 或自定义 path 时不会再扩出很多平行接口


### `ExecutionPlan` 是什么


- 此次运行准备如何放到设备上执行

它不是数值对象，也不是 solver。

它至少描述：

- 用哪些 devices
- 是否开启多 GPU
- 优先沿哪个轴并行
  例如 `k`、`band`
- gpu 的 sharing是什么
- 是否允许 multi-host

一个最小化版本可以长成：

- `device_ids`
- `parallel_axes=("k",)`
- `replicate_fft_axes=True`
- `multi_host=False`

这样它是一个很薄的执行策略对象，而不是很重的框架。


### `RuntimeContext` 至少包含：

跟计算有关的设定，状态，可以提前预计算的部分。

- crystal
- g-grid / r-grid
- ksampling
- basis mask
- ewald data
- pseudopotential objects
- 预计算 projector / beta / augmentation cache
- execution plan


### `SolverSpec`

solver 主要有三个：
- SCFSolver
- DOSolver
- NSCFSolver for band structure.  此处我用的是quantum espresso 的命名，你觉得怎么样，有其他什么意见吗？

SCFSolver 里面可以选择Eigensolver，我们这里都是用的batched logpcg，

我不确定DIIS应该放在哪里


### 如何真正实现 AE / NC / USPP 共用 workflow
这里只写"backend 插件化"还不够，我现在补充一个更具体的方案。

我建议定义一个统一的 ElectronicBackend 接口，至少提供下面这些能力：

build_runtime_terms(ctx)
density_from_state(state, ctx)
total_energy(state, ctx)
hamiltonian_apply(state, ctx, ksampling)
hamiltonian_matrix(state, ctx, ksampling)
extra_forces(state, ctx)
然后：

AllElectronBackend
NormConservingBackend
UltrasoftBackend
分别实现这些接口。

这样 GroundStateWorkflow 就不用知道：

local / nonlocal projector 怎么拼
overlap operator 在哪里插
augmentation charge 怎么处理
它只管：

初始化
调 solver
checkpoint
收敛判断
输出 result
这才是真正避免复制 workflow 的方法。

---

## Comments (from code review)

### 关于 SolverSpec 命名

QE 的 `scf` / `nscf` / `bands` 命名在 DFT 社区是通用术语，用户一看就懂。但 jrystal 还有一个 QE 没有的东西：DirectOpt（直接优化自由能，不走传统 SCF 循环）。所以建议：

- **`scf`** -- 传统 self-consistent field：密度混合 + eigensolver 循环
- **`direct_opt`** -- 直接对 free energy 做梯度下降（当前代码的主要工作模式）
- **`nscf`** -- 固定密度/势，只解特征值（band structure 就是 nscf 的一种）

不建议叫 `DOSolver`，因为 `DO` 在 DFT 社区没有约定俗成的含义，容易和 DOS (density of states) 混淆。叫 `direct_opt` 更清楚。

### 关于 DIIS 的位置

DIIS 是**密度混合**（density mixing）的一种策略，它和 eigensolver（LOBPCG）是并列的两个 SCF 内部组件，不是同一层次：

```
SCFSolver
├── eigensolver: LOBPCG / Davidson
└── density_mixer: DIIS / simple_mixing / kerker
```

所以 DIIS 应该作为 `DensityMixer` 的一种实现，和 `Eigensolver` 平级。当前代码 `scf/diis.py` 已经是独立的 `diis_init` / `diis_update` 函数对，非常适合作为 mixer backend。

### 关于 ElectronicBackend

你列的 6 个方法里，实际第一步只需要 3 个就够了：

1. `build_potentials(ctx)` -- 构建 backend 特有的势（local/nonlocal/augmentation）
2. `total_energy(coeff, occ, ctx)` -- 算总能
3. `hamiltonian_apply(coeff, density, ctx)` -- Hamiltonian 作用于波函数（SCF 的 Hvp）

`density_from_state` 对 AE/NC 来说就是 `pw.density_grid`，不需要抽象；对 USPP 才需要加 augmentation charge，所以可以第二步再加。`hamiltonian_matrix` 和 `extra_forces` 也是后续扩展。先做小再做全。

### 关于 PlaneWaveBasis

`PlaneWaveBasis` 不需要是一个重对象。它只需要封装：
- `freq_mask` (spherical/cubic mask)
- `grid_sizes` (FFT grid 尺寸)
- `num_g` (mask 后的 G-vector 数量)

当前代码里这三个东西散落在 `ctx.freq_mask`、`config.basis.grid_sizes` 和各种 `np.sum(mask)` 里。收到一个小对象里会更清楚。

---

## 详细修改计划

### 目标目录结构

```
jrystal/
├── _src/                   # kernel layer (不动)
├── calc/
│   ├── __init__.py         # public API: run_scf, run_direct_opt, run_bands
│   ├── types.py            # KSampling, PlaneWaveBasis, ExecutionPlan,
│   │                       # EnergyDecomposition, GroundStateResult, BandStructureResult
│   ├── runtime.py          # RuntimeContext, build_runtime_context
│   ├── backend.py          # ElectronicBackend protocol + AllElectronBackend + NormConservingBackend
│   ├── solver_direct_opt.py  # DirectOptSolver
│   ├── solver_scf.py       # SCFSolver (eigensolver + density mixer loop)
│   ├── solver_nscf.py      # NSCFSolver (band structure)
│   ├── convergence.py      # ConvergenceChecker (已有)
│   ├── density_mixing.py   # DIIS, simple_mixing, kerker (从 scf/diis.py 整合)
│   ├── opt_utils.py        # 保留 set_env_params, create_optimizer 等工具
│   └── pre_calc.py         # 保留 SBT 预计算
├── config.py               # (已完成)
├── crystal.py              # (不动)
├── pseudopotential/        # (不动)
└── ...
```

### 分步实施计划

---

#### Step 0: 新建 `types.py` 中缺少的对象

**文件**: `jrystal/calc/types.py`

**新增 `PlaneWaveBasis`**:
```python
@chex_dataclass
class PlaneWaveBasis:
  freq_mask: Bool[Array, "x y z"]
  grid_sizes: tuple[int, ...]
  num_g: int  # = jnp.sum(freq_mask)
```

**新增 `ExecutionPlan`**:
```python
@dataclass
class ExecutionPlan:
  num_devices: int = 1
  parallel_over_k: bool = False
  # 以下字段预留，第一步只用 num_devices + parallel_over_k
  # parallel_axes: tuple[str, ...] = ("k",)
  # replicate_fft_axes: bool = True
  # multi_host: bool = False
```

**修改 `GroundStateResult`**:
- 加 `eigenvalues: Optional[Any] = None` 字段（SCF solver 会产出特征值，DO 不一定有）

**修改 `BandStructureResult`**:
- 加 `density: Optional[Any] = None` 字段（band 可能需要回传密度）

**不动**: `KSampling`, `EnergyDecomposition` 保持现状。

---

#### Step 1: 新建 `backend.py` -- ElectronicBackend

**文件**: `jrystal/calc/backend.py`

定义 Protocol:
```python
class ElectronicBackend(Protocol):
  def build_potentials(self, ctx: RuntimeContext) -> RuntimeContext:
    """构建 backend 特有的势，附加到 ctx 并返回。"""

  def total_energy(self, coeff, occ, ctx: RuntimeContext) -> float:
    """给定波函数系数和 occupation，算总能（不含 ewald）。"""

  def hamiltonian_apply(self, coeff, density, ctx: RuntimeContext):
    """H|psi> -- Hamiltonian 作用于波函数（用于 SCF 的 LOBPCG）。"""
```

实现 `AllElectronBackend`:
```python
class AllElectronBackend:
  def __init__(self, config):
    self.xc = config.method.xc

  def build_potentials(self, ctx):
    # AE 没有额外的势要构建，直接返回
    return ctx

  def total_energy(self, coeff, occ, ctx):
    density = pw.density_grid(coeff, ctx.crystal.vol, occ,
                              k_weights=ctx.ksampling.weights)
    density_reciprocal = pw.density_grid_reciprocal(
      coeff, ctx.crystal.vol, occ, k_weights=ctx.ksampling.weights)
    kin = energy.kinetic(coeff, ctx.g_vec, ctx.ksampling.kpts,
                         kpts_weights=ctx.ksampling.weights, occupation=occ)
    hart = energy.hartree(density_reciprocal, ctx.g_vec, ctx.crystal.vol)
    ext = energy.external(density_reciprocal, ctx.crystal.positions,
                          ctx.crystal.charges, ctx.g_vec, ctx.crystal.vol)
    xc = energy.xc_energy(density, ctx.g_vec, ctx.crystal.vol, self.xc)
    return kin + hart + ext + xc

  def hamiltonian_apply(self, coeff, density, ctx):
    return hamiltonian.hamiltonian_matrix_trace(...)  # 复用现有 _src
```

实现 `NormConservingBackend`:
```python
class NormConservingBackend:
  def __init__(self, config):
    self.xc = config.method.xc

  def build_potentials(self, ctx):
    # 把当前 runtime.py 中的 _build_normcons_potentials 逻辑搬过来
    # 返回 ctx（已附加 potential_local, potential_nonlocal）
    ...

  def total_energy(self, coeff, occ, ctx):
    # 复制当前 calc_ground_state_energy_normcons.py 中的 total_energy 逻辑
    ...

  def hamiltonian_apply(self, coeff, density, ctx):
    # 复用 scf.py 中的 Hvp_fun 逻辑
    ...
```

**关键点**: 从当前 `calc_ground_state_energy_all_electrons.py:98-120` 和 `calc_ground_state_energy_normcons.py:112-133` 中提取 `total_energy` 闭包，改为 backend 方法。不改 `_src` 层代码。

---

#### Step 2: 修改 `runtime.py` -- 让 RuntimeContext 使用 backend

**文件**: `jrystal/calc/runtime.py`

修改 `RuntimeContext`：
```python
@dataclass
class RuntimeContext:
  crystal: Crystal
  g_vec: Float[Array, "x y z 3"]
  r_vec: Float[Array, "x y z 3"]
  ksampling: KSampling
  basis: PlaneWaveBasis          # 新增
  ewald_energy: float
  execution: ExecutionPlan       # 新增
  pseudopotential: Optional[object] = None
  potential_local: Optional[object] = None
  potential_nonlocal: Optional[object] = None
```

修改 `build_runtime_context()`：
1. 构建 `PlaneWaveBasis` 对象
2. 构建 `ExecutionPlan` 对象
3. 调用 `backend.build_potentials(ctx)` 而不是内联的 `_build_normcons_potentials`
4. 删除 `_build_normcons_potentials`（逻辑移到 `NormConservingBackend.build_potentials`）

```python
def build_runtime_context(config, *, mode="mesh", backend=None):
  crystal = create_crystal(config)
  g_vec, r_vec, ksampling = create_grids(config, crystal=crystal, ...)
  freq_mask = create_freq_mask(config, crystal=crystal)
  basis = PlaneWaveBasis(freq_mask=freq_mask, grid_sizes=..., num_g=...)
  execution = ExecutionPlan(
    num_devices=len(jax.devices()),
    parallel_over_k=config.execution.parallel_over_k_mesh,
  )
  ew = get_ewald_coulomb_repulsion(config, crystal=crystal, g_vector_grid=g_vec)

  ctx = RuntimeContext(
    crystal=crystal, g_vec=g_vec, r_vec=r_vec,
    ksampling=ksampling, basis=basis,
    ewald_energy=ew, execution=execution,
  )

  if backend is not None:
    ctx = backend.build_potentials(ctx)

  return ctx
```

---

#### Step 3: 新建 `solver_direct_opt.py`

**文件**: `jrystal/calc/solver_direct_opt.py`

从当前 `calc_ground_state_energy_all_electrons.py` 和 `calc_ground_state_energy_normcons.py` 中提取公共的 optimization loop：

```python
def run_direct_opt(config, ctx, backend) -> GroundStateResult:
  """Direct optimization of free energy via gradient descent."""
  key = jax.random.PRNGKey(config.execution.seed)

  num_electrons = _get_num_electrons(ctx, backend)
  num_bands = ceil(num_electrons / 2) + config.occupation.empty_bands
  num_kpts = ctx.ksampling.kpts.shape[0]

  # --- execution setup (mesh/sharding) ---
  mesh, sharding = _build_sharding(ctx.execution)

  # --- init params ---
  occ_fn = occupation.get_occupation_fn(num_electrons, ...)
  params_pw = pw.param_init(key, num_bands, num_kpts, ctx.basis.freq_mask, ...)
  params_occ = occupation.params_init(num_bands, num_kpts)
  params = {"pw": params_pw, "occ": params_occ}
  optimizer = create_optimizer(config)
  opt_state = optimizer.init(params)

  # --- energy function (delegates to backend) ---
  def free_energy(params_pw, params_occ):
    coeff = pw.coeff(params_pw, ctx.basis.freq_mask, sharding=sharding)
    occ = occ_fn(params_occ)
    e = backend.total_energy(coeff, occ, ctx)
    return e, e

  # --- training loop ---
  convergence_checker = create_convergence_checker(config)
  with mesh:
    @jax.jit
    def update(params, opt_state):
      loss = lambda x: free_energy(x["pw"], x["occ"])
      (loss_val, etot), grad = jax.value_and_grad(loss, has_aux=True)(params)
      updates, opt_state = optimizer.update(grad, opt_state)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss_val, etot

    for i in range(config.solver.epoch):
      params, opt_state, loss_val, etot = update(params, opt_state)
      if convergence_checker.check(etot):
        break

  # --- 组装 result ---
  return _build_result(config, ctx, backend, params, occ_fn, converged)
```

**关键区别**: 这个函数对 AE 和 NC 完全通用，因为 physics 差异全在 `backend.total_energy()` 里。

对比当前代码（两个文件各 ~250 行，80% 相同），新版只有一个 ~100 行的函数。

---

#### Step 4: 新建 `solver_scf.py`

**文件**: `jrystal/calc/solver_scf.py`

从 `scf/scf.py` 脚本中提取，改造为可调用函数:

```python
def run_scf(config, ctx, backend) -> GroundStateResult:
  """Traditional SCF loop: diagonalize + mix density."""
  key = jax.random.PRNGKey(config.execution.seed)
  num_electrons = _get_num_electrons(ctx, backend)
  num_bands = ...

  # --- init ---
  coeff = _init_random_coeff(key, num_bands, ctx)
  evals = jnp.zeros([...])
  occ = _fixed_occupation(evals, num_electrons)
  dens = pw.density_grid(coeff, ctx.crystal.vol, occ, ...)
  diis_state = diis_init(max_hist=8, density_shape=dens.shape, dtype=dens.dtype)

  # --- SCF loop ---
  for i in range(config.solver.scf_max_iter):
    # 1. diagonalize: LOBPCG
    coeff_new, evals_new = _diagonalize(
      coeff, dens, ctx, backend,
      lobpcg_max_iter=config.solver.lobpcg_max_iter,
    )

    # 2. update occupation
    occ = _compute_occupation(evals_new, num_electrons, config)

    # 3. new density
    dens_new = pw.density_grid(coeff_new, ctx.crystal.vol, occ, ...)

    # 4. convergence check
    if _converged(dens_new, dens, tol):
      break

    # 5. density mixing (DIIS)
    dens_error = dens_new - dens
    diis_state, dens_mixed = diis_update(diis_state, dens_new, dens_error)
    dens = _simple_mixing(dens_mixed, dens, beta=config.solver.mixing_beta)
    coeff = coeff_new

  return _build_result(config, ctx, backend, ...)
```

**内部函数 `_diagonalize`**:
```python
def _diagonalize(coeff, density, ctx, backend, lobpcg_max_iter):
  """一步 LOBPCG 对角化。"""
  def hvp(c):
    return backend.hamiltonian_apply(c, density, ctx)

  eigval, evec = batched_lobpcg(
    matmul=hvp,
    k=num_bands,
    v0=coeff,
    which="smallest",
    maxit=lobpcg_max_iter,
  )
  return evec, eigval
```

---

#### Step 5: 新建 `solver_nscf.py`

**文件**: `jrystal/calc/solver_nscf.py`

从当前 `calc_band_structure_all_electrons.py` 和 `calc_band_structure_normcons.py` 中提取。band structure 就是 nscf 的一种：固定密度，在 k-path 上算特征值。

```python
def run_nscf(config, ctx, backend,
             ground_state_result: GroundStateResult) -> BandStructureResult:
  """Non-self-consistent calculation along a k-path."""
  density = ground_state_result.density

  # 对 AE: 在每个 k 点 minimize hamiltonian trace
  # 对 NC: 同上但要构建每个 k 点的 nonlocal potential

  # ... (提取当前 band calc 文件中的 optimize_eigenvalues 逻辑)

  return BandStructureResult(...)
```

这个文件的内部复杂度较高（pmap、lax.scan、fine-tuning），但外部接口简单。

---

#### Step 6: 修改 `calc/__init__.py` -- 统一入口

**文件**: `jrystal/calc/__init__.py`

```python
from .backend import AllElectronBackend, NormConservingBackend
from .runtime import build_runtime_context
from .solver_direct_opt import run_direct_opt
from .solver_scf import run_scf
from .solver_nscf import run_nscf
from .types import GroundStateResult, BandStructureResult

def _get_backend(config):
  if config.method.use_pseudopotential:
    if config.method.pseudopotential_type in ("nc", "normcons", "normconserving"):
      return NormConservingBackend(config)
    raise NotImplementedError(...)
  return AllElectronBackend(config)

def energy(config) -> GroundStateResult:
  """Run a ground-state calculation (auto-selects solver + backend)."""
  backend = _get_backend(config)
  ctx = build_runtime_context(config, mode="mesh", backend=backend)
  if config.solver.type == "scf":
    return run_scf(config, ctx, backend)
  else:
    return run_direct_opt(config, ctx, backend)

def band(config, ground_state_result=None) -> BandStructureResult:
  """Run a band-structure calculation."""
  if ground_state_result is None:
    ground_state_result = energy(config)
  backend = _get_backend(config)
  ctx = build_runtime_context(config, mode="path", backend=backend)
  return run_nscf(config, ctx, backend, ground_state_result)
```

用户只需要：
```python
import jrystal as jr
config = jr.config.get_config("config.yaml")
result = jr.calc.energy(config)
band_result = jr.calc.band(config, result)
```

---

#### Step 7: 修改 `main.py`

```python
if args.mode == "energy":
    jr.calc.energy(config)
elif args.mode == "band":
    jr.calc.band(config)
```

不再需要 `if config.method.use_pseudopotential:` 分支——backend 选择已经在 `calc.energy()` 内部自动完成。

---

#### Step 8: 迁移 `density_mixing.py`

**文件**: `jrystal/calc/density_mixing.py`

从 `jrystal/scf/diis.py` 搬过来，加上 simple_mixing 和 kerker preconditioning:

```python
# 从 scf/diis.py 直接搬入
from jrystal.scf.diis import diis_init, diis_update

def simple_mixing(new_density, old_density, beta=0.7):
  return new_density * beta + old_density * (1 - beta)

def kerker_preconditioner(g_vec, freq_mask):
  """Kerker preconditioner for density mixing."""
  eff_g = g_vec.at[freq_mask].get()
  g2 = jnp.sum(eff_g**2, axis=-1)
  return g2 / (1 + g2)
```

---

#### Step 9: 清理旧文件

删除（或移到 `_archive`）：
- `calc/calc_ground_state_energy_all_electrons.py`
- `calc/calc_ground_state_energy_normcons.py`
- `calc/calc_band_structure_all_electrons.py`
- `calc/calc_band_structure_normcons.py`

这些文件的逻辑已经被拆分到 `backend.py` + `solver_*.py`。

---

#### Step 10: 补测试

- `tests/smoke/test_backend.py` -- 测 AE / NC backend 的 `total_energy` 对 baseline 不 regress
- `tests/smoke/test_solver.py` -- 测 `run_direct_opt` + `run_scf` 的最小运行
- `tests/smoke/test_public_api.py` -- 测 `jr.calc.energy()` / `jr.calc.band()` 入口
- 复用 `tests/reference/baseline_ae_diamond.yaml` 和 `tests/reference/baseline_nc_diamond.yaml` 做回归

---

### 执行顺序

严格按 Step 0 → 10 顺序。每一步的前置依赖：

```
Step 0 (types.py 补对象)
  → Step 1 (backend.py)
    → Step 2 (runtime.py 改造)
      → Step 3 (solver_direct_opt.py)  -- 可和 Step 4 并行
      → Step 4 (solver_scf.py)         -- 可和 Step 3 并行
      → Step 5 (solver_nscf.py)
        → Step 6 (calc/__init__.py 统一入口)
          → Step 7 (main.py)
Step 8 (density_mixing.py) -- 可和 Step 3-5 并行
Step 9 (清理旧文件) -- 所有 solver 就绪后
Step 10 (补测试) -- 每步都补，最后汇总验证
```

### 各文件改动 cheatsheet

| 文件 | 动作 | 来源 |
|------|------|------|
| `calc/types.py` | 修改 | 加 `PlaneWaveBasis`, `ExecutionPlan`，改 result 字段 |
| `calc/backend.py` | 新建 | `ElectronicBackend` Protocol + AE/NC 两个实现 |
| `calc/runtime.py` | 修改 | `RuntimeContext` 加 `basis`/`execution` 字段，`build_runtime_context` 接 backend |
| `calc/solver_direct_opt.py` | 新建 | 从 `calc_ground_state_energy_*.py` 提取公共 loop |
| `calc/solver_scf.py` | 新建 | 从 `scf/scf.py` 脚本改造 |
| `calc/solver_nscf.py` | 新建 | 从 `calc_band_structure_*.py` 提取 |
| `calc/density_mixing.py` | 新建 | 从 `scf/diis.py` 搬入 + 加 simple_mixing/kerker |
| `calc/__init__.py` | 修改 | `energy()` + `band()` 统一入口 |
| `main.py` | 修改 | 简化为调 `jr.calc.energy` / `jr.calc.band` |
| `calc/calc_ground_state_energy_*.py` | 删除/archive | 被 backend + solver 替代 |
| `calc/calc_band_structure_*.py` | 删除/archive | 被 solver_nscf 替代 |
| `calc/convergence.py` | 不动 | |
| `calc/opt_utils.py` | 小改 | 可能移除不再需要的 helper |
| `calc/pre_calc.py` | 不动 | |
| `_src/` | 不动 | kernel 层不变 |
