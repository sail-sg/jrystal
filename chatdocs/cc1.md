# Jrystal Refactoring Step 1: Executable Plan

Branch: `refactor/step-1` (from `scf/refactor`)

## 0. cc0.md 中问题的决策

在写计划之前，先把 cc0.md 中的 8 个问题全部定下来。

**Q1: k-point weights** -- 确认是 physics bug。`_src` 层的 `energy.kinetic()`、`pw.density_grid()`、`pw.density_grid_reciprocal()` 都已经接受 `k_weights` 参数，但 `calc` 层从未传入。`create_grids()` 甚至没有捕获 `k_vectors()` 返回的 weights。对于多 k-point 计算，当前结果是错误的（所有 k-point 被当作等权重）。本次重构必须修复。

**Q2: SCF 定位** -- 第一步只收敛 DirectOpt workflow。`jrystal/scf/` 下的脚本保持原样不动，不做模块化，也不 archive。第二步再处理。

**Q3: `pseudopotential_type` 字段** -- config 的 `default_config` 缺失这个字段但代码在用。本次统一 config 时补齐。

**Q4: `chex.dataclass` vs `dataclasses.dataclass`** -- 需要过 `jax.jit` 边界的对象用 `chex.dataclass`（如 `KSampling`、`RuntimeContext`）。纯结果容器用标准 `dataclass`（如 `GroundStateResult`）。

**Q5: `calc` vs `workflows`** -- 保留 `jrystal/calc/`，不引入新的 `workflows` module。新的 types 和 runtime 直接放在 `jrystal/calc/` 下。

**Q6: calc 文件保留策略** -- 见下方 WP1 详细说明。

**Q7: k-weights 在 pipeline 中的消费层** -- weights 应在 `density_grid()`/`density_grid_reciprocal()` 和 `energy.kinetic()` 层面传入。occupation 层不管 weights。这意味着 `KSampling` 对象应该在 workflow 的 energy 计算函数中被消费。

**Q8: branch** -- 已创建 `refactor/step-1`。

---

## 1. 实施顺序概览

```
WP0  记录 baseline                       (0.5h)
WP1  清理 calc/ + 去掉 jaxopt + 修 import (2h)
WP2  统一 config                          (3h)
WP3  引入 KSampling + 修复 k-weights bug  (3h)
WP4  引入 RuntimeContext                   (3h)
WP5  统一 result dataclass                 (2h)
WP6  文档与 examples 最小对齐              (1h)
```

每个 WP 完成后立即补对应的 smoke test，不单独设测试 WP。

---

## WP0: 记录 baseline

### 目标
在改任何代码之前，跑一次当前代码的 canonical benchmark，把数值结果存下来，作为重构期间的回归基线。

### 具体做什么
1. 用当前 `config.yaml` 跑 all-electron Si diamond（因为 normcons 路径的 `calc/__init__.py` 会因 jaxopt 报错，先跑能跑的那个）
2. 记录 total energy、各 energy component 到 `tests/reference/baseline_ae_si_diamond.yaml`
3. 如果 normcons 能跑（手动绕过 import），也记一份 `tests/reference/baseline_nc_si_diamond.yaml`

### 验收
- `tests/reference/` 目录下有至少一份 baseline 数据文件

---

## WP1: 清理 calc/ + 去掉 jaxopt + 修 import

### 目标
让 `import jrystal.calc` 能正常工作，清理掉不会在第一阶段维护的文件。

### 具体改动

#### 1.1 去掉 jaxopt 路线
- 把 `calc/__init__.py` 中的 `energy_normcons` 从 import `normcons_cg` 改为 import `normcons`（即 optax 版本）
- 把 `calc_ground_state_energy_normcons_cg.py` 移到 `calc/_archive/`（不删除，保留参考）

#### 1.2 archive 不维护的 calc 文件
移到 `calc/_archive/`：
- `calc_ground_state_energy_normcons_cg.py` (jaxopt)
- `calc_ground_state_energy_normcons_psgd.py` (实验性)
- `calc_ground_state_energy_ultrasoft.py` (第二阶段)
- `calc_ground_state_energy_hartree_fock.py` (非主线)
- `calc_band_structure_ultrasoft.py` (第二阶段)
- `preconditioned_sgd.py` (空文件)
- `geo_opt_all_electron.py` (空文件)

保留在 `calc/`：
- `calc_ground_state_energy_all_electrons.py` (验证 backend)
- `calc_ground_state_energy_normcons.py` (主线)
- `calc_band_structure_all_electrons.py` (验证)
- `calc_band_structure_normcons.py` (主线)
- `opt_utils.py`
- `pre_calc.py`
- `convergence.py`

原有的 `_backup/` 保持不动。

#### 1.3 修 `calc/__init__.py`
```python
from .calc_ground_state_energy_all_electrons import calc as energy_all_electrons
from .calc_band_structure_all_electrons import calc as band_all_electrons
from .calc_ground_state_energy_normcons import calc as energy_normcons
from .calc_band_structure_normcons import calc as band_normcons

__all__ = [
    "energy_all_electrons",
    "band_all_electrons",
    "energy_normcons",
    "band_normcons",
]
```

#### 1.4 修 `jrystal/__init__.py`
加入 `calc` 的导出：
```python
from . import calc
```
加到 `__all__` 列表中。

#### 1.5 修 `main.py`
- 去掉 HF 分支（已 archive）
- 保留 energy / band 两个 mode
- 保留 all-electron / normcons 分支

### 补 smoke test
新建 `tests/smoke/test_import.py`：
```python
def test_import_jrystal():
    import jrystal

def test_import_calc():
    import jrystal.calc

def test_calc_has_entries():
    import jrystal.calc as calc
    assert hasattr(calc, 'energy_normcons')
    assert hasattr(calc, 'energy_all_electrons')
    assert hasattr(calc, 'band_normcons')
    assert hasattr(calc, 'band_all_electrons')
```

### 验收
- `import jrystal.calc` 不报错
- `python main.py -h` 能正常输出
- smoke test 通过

---

## WP2: 统一 config

### 目标
config 字段分块、补齐缺失字段、加 validation。

### 设计决策
- **继续用 `ml_collections.ConfigDict`**，不迁移到 pydantic。原因：改配置框架不是这一步的目标，风险大收益小。
- 用手写 validation 函数处理 unknown fields。
- 支持 flat YAML 自动迁移到 nested 结构（策略 A）。

### 具体改动

#### 2.1 重写 `jrystal/config.py`

新的 `default_config` 分块：
```python
default_config = {
    "schema_version": 1,

    # system
    "system": {
        "crystal": "diamond",
        "crystal_file_path": None,  # 修正 typo: crystal_file_path_path
        "spin": 0,
        "spin_restricted": True,
    },

    # method
    "method": {
        "xc": "lda_x",
        "use_pseudopotential": False,
        "pseudopotential_type": "nc",  # 补齐缺失字段
        "pseudopotential_file_dir": None,
    },

    # basis
    "basis": {
        "freq_mask_method": "spherical",
        "cutoff_energy": 100,
        "grid_sizes": 48,
    },

    # ksampling
    "ksampling": {
        "k_grid_sizes": [4, 4, 4],
        "symmetry_reduction": True,  # 新增，对应当前硬编码行为
    },

    # solver
    "solver": {
        "type": "direct_opt",  # "direct_opt" | "scf"
        "optimizer": "adam",
        "optimizer_args": {"learning_rate": 0.01, "b1": 0.9, "b2": 0.99},
        "scheduler": None,
        "epoch": 10000,
        "convergence_window_size": 20,
        "convergence_condition": 1e-6,
    },

    # occupation
    "occupation": {
        "method": "uniform",
        "smearing": 0.0,
        "empty_bands": 20,
    },

    # ewald
    "ewald": {
        "eta": 0.1,
        "cutoff": 2e4,
    },

    # band
    "band": {
        "empty_bands": None,  # defaults to occupation.empty_bands
        "k_path_special_points": None,
        "num_kpoints": 64,
        "k_path_file": None,
        "epoch": 5000,
        "fine_tuning": True,
        "fine_tuning_epoch": 300,
    },

    # execution
    "execution": {
        "seed": 123,
        "parallel_over_k": False,
        "jax_enable_x64": True,
        "jax_debug_nans": False,
        "verbose": True,
        "eps": 1e-8,
    },

    # io
    "io": {
        "save_dir": None,
    },
}
```

#### 2.2 写 `_migrate_flat_config()` 函数
接收旧格式 flat dict，自动映射到新的 nested 结构。这样旧的 `config.yaml` 文件不会立刻失效。

```python
def _migrate_flat_config(flat: dict) -> dict:
    """Convert legacy flat config to nested schema v1."""
    # 映射表...
    # 如果已经有 schema_version，直接返回
    # 否则按字段名映射到对应的 nested group
```

#### 2.3 写 `validate_config()` 函数
- 检查 unknown fields 并 warning
- 检查关键字段类型

#### 2.4 更新 `config.yaml`
改成 nested 格式，作为新格式的示例。

#### 2.5 更新 `opt_utils.py`
所有 `config.xxx` 访问改为 `config.system.xxx` / `config.method.xxx` 等。这是这一步工作量最大的部分。涉及文件：
- `opt_utils.py`
- `calc_ground_state_energy_all_electrons.py`
- `calc_ground_state_energy_normcons.py`
- `calc_band_structure_all_electrons.py`
- `calc_band_structure_normcons.py`
- `main.py`

### 补 smoke test
新建 `tests/smoke/test_config.py`：
```python
def test_default_config_loads():
    from jrystal.config import get_config
    config = get_config()
    assert config.schema_version == 1

def test_legacy_config_migrates():
    # 用一个旧格式 dict 测试 migration
    ...

def test_yaml_config_loads():
    from jrystal.config import get_config
    config = get_config("config.yaml")
    assert config.schema_version == 1
```

### 验收
- 新格式 `config.yaml` 能正常加载
- 旧格式 flat yaml 能自动迁移
- 所有 calc 函数能正常读取新 config 的字段

---

## WP3: 引入 KSampling + 修复 k-weights bug

### 目标
1. 统一 k-point mesh/path 的 contract
2. 修复 k-point weights 从未被传入 energy/density 计算的 bug

### 具体改动

#### 3.1 新建 `jrystal/calc/types.py`

```python
from chex import dataclass
from typing import Optional, Literal
from jaxtyping import Array, Float

@dataclass
class KSampling:
    """Unified k-point sampling object."""
    mode: str  # "mesh" | "path"
    kpts: Float[Array, "kpt 3"]
    weights: Float[Array, " kpt"]
    labels: Optional[list] = None      # for path mode
    segments: Optional[list] = None    # for path mode
```

#### 3.2 修改 `opt_utils.py` 的 `create_grids()`

```python
def create_grids(config):
    crystal = create_crystal(config)
    grid_sizes = proper_grid_size(config.basis.grid_sizes)
    k_grid_sizes = proper_grid_size(config.ksampling.k_grid_sizes)
    g_vec = g_vectors(crystal.cell_vectors, grid_sizes)
    r_vec = r_vectors(crystal.cell_vectors, grid_sizes)
    kpts, k_weights = k_vectors(
        crystal.cell_vectors, k_grid_sizes,
        symmetry_reduction=config.ksampling.symmetry_reduction,
        scaled_positions=crystal.scaled_positions,
        charges=crystal.charges,
    )
    ksampling = KSampling(mode="mesh", kpts=kpts, weights=k_weights)
    return g_vec, r_vec, ksampling
```

#### 3.3 修复所有 calc 文件中的 k-weights bug

以 `calc_ground_state_energy_normcons.py` 为例，核心改动：

```python
# 之前:
g_vec, r_vec, k_vec = create_grids(config)

# 之后:
g_vec, r_vec, ksampling = create_grids(config)
k_vec = ksampling.kpts
k_weights = ksampling.weights
```

然后在 total_energy 函数内：
```python
# 之前:
density = pw.density_grid(coeff, crystal.vol, occ)
density_reciprocal = pw.density_grid_reciprocal(coeff, crystal.vol, occ)
kinetic = energy.kinetic(g_vec, k_vec, coeff, occ)

# 之后:
density = pw.density_grid(coeff, crystal.vol, occ, k_weights=k_weights)
density_reciprocal = pw.density_grid_reciprocal(coeff, crystal.vol, occ, k_weights=k_weights)
kinetic = energy.kinetic(g_vec, k_vec, coeff, occ, kpts_weights=k_weights)
```

同样的修改应用到：
- `calc_ground_state_energy_all_electrons.py`
- `calc_ground_state_energy_normcons.py`
- `calc_band_structure_normcons.py`
- `calc_band_structure_all_electrons.py`
- 以及每个文件末尾"END OF OPTIMIZATION"后的 energy decomposition 代码

#### 3.4 band path 也走 KSampling

`calc_band_structure_normcons.py` 中的 k_path 也封装为 `KSampling(mode="path", ...)`。

### 补 smoke test
`tests/smoke/test_ksampling.py`：
```python
def test_mesh_ksampling_has_weights():
    ksampling = create_grids(config)[2]
    assert ksampling.mode == "mesh"
    assert ksampling.weights.shape[0] == ksampling.kpts.shape[0]
    assert abs(ksampling.weights.sum() - 1.0) < 1e-6

def test_path_ksampling():
    ks = KSampling(mode="path", kpts=..., weights=..., labels=["G", "X"])
    assert ks.mode == "path"
```

### 验收
- `create_grids()` 返回 `KSampling` 对象
- k-weights 被正确传入所有 energy/density 计算
- 现有 baseline 数值不 regress（对于 Gamma-only 或 uniform k-mesh，加 weights 后结果应该不变）

---

## WP4: 引入 RuntimeContext

### 目标
把 workflow 初始化阶段的预处理结果集中到一个对象中，避免每个 calc 文件各自重复构建。

### 具体改动

#### 4.1 新建 `jrystal/calc/runtime.py`

```python
from chex import dataclass
from typing import Optional
from jaxtyping import Array, Float, Bool

@dataclass
class RuntimeContext:
    """Collected runtime state for a calculation."""
    crystal: Crystal
    g_vec: Float[Array, "x y z 3"]
    r_vec: Float[Array, "x y z 3"]
    ksampling: KSampling
    freq_mask: Bool[Array, "x y z"]
    ewald_energy: float
    # pseudopotential fields (None for all-electron)
    pseudopotential: Optional[object] = None
    potential_local: Optional[Array] = None
    potential_nonlocal: Optional[Array] = None
```

#### 4.2 新建 `build_runtime_context()` 函数

在 `opt_utils.py` 或 `runtime.py` 中：
```python
def build_runtime_context(config) -> RuntimeContext:
    """One-shot initialization of all runtime data."""
    crystal = create_crystal(config)
    g_vec, r_vec, ksampling = create_grids(config)
    freq_mask = create_freq_mask(config)
    ew = get_ewald_coulomb_repulsion(config)

    pp = None
    pot_loc = None
    pot_nl = None
    if config.method.use_pseudopotential:
        pp = create_pseudopotential(config)
        pot_loc = ...  # local potential
        pot_nl = ...   # nonlocal potential (including SBT)

    return RuntimeContext(
        crystal=crystal,
        g_vec=g_vec,
        r_vec=r_vec,
        ksampling=ksampling,
        freq_mask=freq_mask,
        ewald_energy=ew,
        pseudopotential=pp,
        potential_local=pot_loc,
        potential_nonlocal=pot_nl,
    )
```

#### 4.3 重构 calc 文件使用 RuntimeContext

以 `calc_ground_state_energy_normcons.py` 为例：
```python
def calc(config):
    set_env_params(config)
    ctx = build_runtime_context(config)
    # 所有预处理数据从 ctx 中取
    # crystal = ctx.crystal
    # k_vec = ctx.ksampling.kpts
    # ...
```

这一步的目标是让 `calc` 函数的前 80 行初始化代码收敛到一个 `build_runtime_context()` 调用。四个保留的 calc 文件都要改。

#### 4.4 定义最小 ElectronicBackend protocol

在 `jrystal/calc/types.py` 中加一个 Protocol，为第二步的 AE/NC/USPP 统一做准备：
```python
from typing import Protocol

class ElectronicBackend(Protocol):
    def build_potentials(self, ctx: RuntimeContext) -> RuntimeContext:
        """Compute backend-specific potentials and attach to context."""
        ...

    def total_energy(self, params, ctx: RuntimeContext) -> float:
        """Compute total energy given parameters and runtime context."""
        ...
```

第一步不要求所有 calc 文件都实现这个 Protocol，只是把接口定义写出来，让后续重构有锚点。

### 补 smoke test
`tests/smoke/test_runtime.py`：
```python
def test_build_runtime_context_ae():
    config = get_config()  # all-electron default
    ctx = build_runtime_context(config)
    assert ctx.crystal is not None
    assert ctx.ksampling.mode == "mesh"
    assert ctx.pseudopotential is None
```

### 验收
- 四个 calc 文件都通过 `RuntimeContext` 获取预处理数据
- `build_runtime_context()` 是唯一的预处理入口
- baseline 数值不 regress

---

## WP5: 统一 result dataclass

### 目标
calc 函数真正返回 result 对象，而不是裸 density。

### 具体改动

#### 5.1 在 `jrystal/calc/types.py` 中定义统一 result

```python
from dataclasses import dataclass, field

@dataclass
class EnergyDecomposition:
    kinetic: float = 0.0
    hartree: float = 0.0
    xc: float = 0.0
    external: float = 0.0       # all-electron external
    external_local: float = 0.0  # NC local
    external_nonlocal: float = 0.0  # NC nonlocal
    ewald: float = 0.0

@dataclass
class GroundStateResult:
    config: object
    crystal: object
    params_pw: dict
    params_occ: dict
    total_energy: float
    energy_terms: EnergyDecomposition
    converged: bool
    density: object  # jax.Array
    total_energy_history: list = field(default_factory=list)

@dataclass
class BandStructureResult:
    config: object
    crystal: object
    kpath: KSampling
    eigenvalues: object  # jax.Array
    ground_state_energy: float = 0.0
```

#### 5.2 修改所有 calc 函数的返回值

以 normcons 为例：
```python
# 当前: return density
# 改为:
return GroundStateResult(
    config=config,
    crystal=crystal,
    params_pw=params["pw"],
    params_occ=params["occ"],
    total_energy=float(etot + ew),
    energy_terms=EnergyDecomposition(
        kinetic=float(kinetic),
        hartree=float(hartree),
        xc=float(xc),
        external_local=float(external_local),
        external_nonlocal=float(external_nonlocal),
        ewald=float(ew),
    ),
    converged=converged,
    density=density,
)
```

#### 5.3 删除各文件中重复定义的 `GroundStateEnergyOutput`

现在每个 calc 文件都自己定义了一份 `GroundStateEnergyOutput`，统一到 `types.py` 后删除。

#### 5.4 修改 band 计算文件

`calc_band_structure_normcons.py` 需要改为：
- 接收 `GroundStateResult` 而不是旧的 `GroundStateEnergyOutput`
- 返回 `BandStructureResult`

### 补 smoke test
`tests/smoke/test_result.py`：
```python
def test_ground_state_result_fields():
    result = GroundStateResult(...)
    assert hasattr(result, 'total_energy')
    assert hasattr(result, 'energy_terms')
    assert hasattr(result, 'converged')
    assert hasattr(result, 'density')
```

### 验收
- 所有 calc 函数返回 `GroundStateResult` 或 `BandStructureResult`
- band workflow 能正确消费 ground-state result
- `GroundStateEnergyOutput` 不再存在于 calc 文件中

---

## WP6: 文档与 examples 最小对齐

### 目标
让 quickstart 和一个 minimal example 与重构后的 API 一致。

### 具体改动

#### 6.1 更新 `docs/quickstart.md`
改为使用新的 nested config 和 `jr.calc.energy_normcons(config)` 调用。

#### 6.2 新建 `examples/minimal_ground_state.py`
一个能跑通的最小示例：
```python
import jrystal as jr

config = jr.config.get_config("config.yaml")
result = jr.calc.energy_all_electrons(config)
print(f"Total energy: {result.total_energy:.6f} Ha")
print(f"Converged: {result.converged}")
```

#### 6.3 新建 `examples/minimal_band_structure.py`
一个 band 计算示例。

### 验收
- quickstart 中的代码能实际运行
- examples 目录下有可跑通的脚本

---

## 测试策略总结

### 测试框架选择
继续用 `absltest`（项目已有 49 个 absltest 测试），不引入 pytest。新的 smoke test 也用 absltest。

### 测试目录
```
tests/
  smoke/           # 新建，WP1-WP5 的 smoke test
    test_import.py
    test_config.py
    test_ksampling.py
    test_runtime.py
    test_result.py
  reference/       # 新建，baseline 数据
    baseline_ae_si_diamond.yaml
```

现有 `_src/*_test.py` 保持不动，不迁移。

### CI
在 `.github/workflows/` 中新增 `test.yml`，只跑：
- `tests/smoke/`
- `jrystal/_src/*_test.py` 中不需要 GPU 的 fast unit tests

---

## 文件影响范围

### 新建文件
- `jrystal/calc/types.py` -- KSampling, RuntimeContext, result dataclasses, ElectronicBackend protocol
- `jrystal/calc/runtime.py` -- build_runtime_context()
- `tests/smoke/test_import.py`
- `tests/smoke/test_config.py`
- `tests/smoke/test_ksampling.py`
- `tests/smoke/test_runtime.py`
- `tests/smoke/test_result.py`
- `tests/reference/baseline_*.yaml`
- `examples/minimal_ground_state.py`
- `examples/minimal_band_structure.py`
- `.github/workflows/test.yml`

### 修改文件
- `jrystal/__init__.py` -- 加 calc 导出
- `jrystal/config.py` -- nested config, migration, validation
- `jrystal/calc/__init__.py` -- 改 import 来源
- `jrystal/calc/opt_utils.py` -- create_grids 返回 KSampling, config 字段路径
- `jrystal/calc/calc_ground_state_energy_all_electrons.py` -- RuntimeContext + result + k-weights
- `jrystal/calc/calc_ground_state_energy_normcons.py` -- 同上
- `jrystal/calc/calc_band_structure_all_electrons.py` -- 同上
- `jrystal/calc/calc_band_structure_normcons.py` -- 同上
- `jrystal/calc/pre_calc.py` -- config 字段路径
- `jrystal/calc/convergence.py` -- config 字段路径
- `main.py` -- 去 HF, config 字段路径
- `config.yaml` -- nested 格式
- `docs/quickstart.md` -- 对齐新 API

### 移动文件 (到 `calc/_archive/`)
- `calc_ground_state_energy_normcons_cg.py`
- `calc_ground_state_energy_normcons_psgd.py`
- `calc_ground_state_energy_ultrasoft.py`
- `calc_ground_state_energy_hartree_fock.py`
- `calc_band_structure_ultrasoft.py`
- `preconditioned_sgd.py`
- `geo_opt_all_electron.py`

### 不动的文件
- `jrystal/_src/` 下所有 kernel 文件 -- 不改数学实现
- `jrystal/scf/` -- 第二步再处理
- `jrystal/pseudopotential/` -- 内部实现不动，只是被 RuntimeContext 调用
- `jrystal/sbt/` -- 不动
- 所有 proxy module (`jrystal/energy.py`, `jrystal/pw.py` 等) -- 保持原样
- `jrystal/_src/*_test.py` -- 现有测试不动

---

## 关于 proxy module 层的说明

调查发现所有 11 个 proxy module（`jrystal/energy.py`, `jrystal/pw.py` 等）都是纯 re-export，`calc` 层完全绕过它们直接 import `_src`。

**本次不动它们**。它们对外部用户仍有意义（`jr.grid.g_vectors()` 比 `jr._src.grid.g_vectors()` 好看），而且改它们对重构没有帮助、徒增风险。如果后续要清理，可以作为单独的一步。
