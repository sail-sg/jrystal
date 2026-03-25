# 用户交互设计 + 实施计划

## 一、设计理念

QE 的用户体验好在两点：
1. **命令就是计算类型**: `pw.x`, `bands.x`, `pp.x` — 用户不需要 `-m energy` 这种 flag
2. **输入文件只写必要字段**: 用户不需要知道所有 50 个参数，只写 crystal + xc + cutoff 就能跑

jrystal 应该学这两点，但用现代 CLI 的方式做。

---

## 二、CLI 设计

### 当前（要改掉）

```bash
jrystal -m energy -c config.yaml
jrystal -m band -c config.yaml
```

### 新设计

```bash
# subcommand = 计算类型
jrystal scf config.yaml          # SCF ground state
jrystal direct-opt config.yaml   # Direct optimization ground state
jrystal band config.yaml         # Band structure (auto-runs ground state if needed)

# config file 是 positional argument，不是 -c flag
# 省略时默认 config.yaml
jrystal scf                      # 等价于 jrystal scf config.yaml

# 通用 override：任何 config 字段都可以在命令行覆盖
jrystal scf config.yaml --basis.cutoff_energy=200 --solver.epoch=5000

# 快捷 alias
jrystal energy config.yaml       # = direct-opt (默认 solver)
```

### Subcommand 列表

| 命令 | 作用 | 对应 solver |
|------|------|------------|
| `jrystal scf` | SCF ground state | `run_scf` |
| `jrystal direct-opt` | Direct optimization ground state | `run_direct_opt` |
| `jrystal energy` | Ground state (根据 config 中 `solver.type` 选择) | auto |
| `jrystal band` | Band structure | `run_nscf` (先跑 energy 如果没有) |

---

## 三、Input 文件设计

### 原则：用户只写必要字段，其他全用默认值

QE 的 `pw.in` 只需要写 `ibrav`, `nat`, `ntyp`, `ecutwfc`, `CELL_PARAMETERS`, `ATOMIC_SPECIES`, `ATOMIC_POSITIONS`, `K_POINTS` 这几个必要项。其他几百个参数都有默认值。

jrystal 也应该这样。用户的最小 input file:

### 最小 all-electron input

```yaml
# si_ae.yaml -- 7 行搞定
system:
  crystal_file_path: geometry/diamond.xyz

method:
  xc: lda_x

basis:
  cutoff_energy: 50
  grid_sizes: 24
```

跑法: `jrystal energy si_ae.yaml`

所有没写的字段（ksampling, solver, occupation, ewald, execution...）全用 `default_config` 的值。

### 最小 norm-conserving input

```yaml
# si_nc.yaml -- 10 行
system:
  crystal_file_path: geometry/diamond.xyz

method:
  xc: lda_x
  use_pseudopotential: true
  pseudopotential_type: nc

basis:
  cutoff_energy: 50
  grid_sizes: 24
```

### 最小 band structure input

```yaml
# si_band.yaml -- 在 ground state input 基础上加 band section
system:
  crystal_file_path: geometry/diamond.xyz

method:
  xc: lda_x

basis:
  cutoff_energy: 50
  grid_sizes: 24

band:
  k_path_special_points: LGXL
```

跑法: `jrystal band si_band.yaml`

### 现在的 config.yaml 有什么问题

当前 `config.yaml` 有 65 行，包含所有字段。这不应该是用户需要写的——这是"完整参数参考"，不是"用户 input"。

应该把它拆成：
- **用户 input 文件**：只写必要字段（5-15 行）
- **default_config**：代码里的完整默认值（已有）
- **参考文档**：所有字段及其说明（docs 里）

---

## 四、Config 字段简化

### 可以去掉或重命名的字段

| 当前字段 | 问题 | 建议 |
|---------|------|------|
| `system.crystal` | 和 `crystal_file_path` 重叠，且只支持内置 geometry | 去掉。统一用 `crystal_file_path`，内置 geometry 用 `geometry/diamond.xyz` 相对路径 |
| `method.use_pseudopotential` | bool + pseudopotential_type 两个字段控制一件事 | 合并：`pseudopotential_type: none` 表示 AE，`nc` 表示 NC |
| `basis.freq_mask_method` | 几乎永远是 spherical | 保留但默认 spherical，用户基本不用写 |
| `occupation.method` | "uniform" 这个名字不是 DFT 术语 | 考虑后续改，但现在先不动 |

### `system.crystal` vs `crystal_file_path` 统一方案

当前逻辑：如果 `system.crystal` 不是 None，就去 `geometry/{crystal}.xyz` 找；否则用 `crystal_file_path`。

问题：两个字段控制同一件事，容易混淆。

建议统一为一个字段 `system.structure`：
- 如果是相对路径或绝对路径 → 直接读文件
- 如果是内置名（`diamond`, `si`, `al_bcc`...）→ 去 `geometry/` 下找

```yaml
system:
  structure: diamond         # 内置名
  # 或
  structure: my_crystal.xyz  # 文件路径
```

### `method.pseudopotential_type` 统一方案

```yaml
method:
  xc: lda_x
  pseudopotential: none    # all-electron (默认)
  # pseudopotential: nc    # norm-conserving
  # pseudopotential: us    # ultrasoft (未来)
  pseudopotential_dir: /path/to/pp/  # 可选
```

去掉 `use_pseudopotential` bool，一个字段搞定。

---

## 五、详细实施计划

### Step A: 重写 CLI (`main.py`)

**文件**: `main.py`

从 `argparse` 单一 parser 改为 subcommand 结构：

```python
import argparse
import jrystal as jr


def main():
  parser = argparse.ArgumentParser(
    prog="jrystal",
    description="JAX-based Differentiable DFT Framework",
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  # --- jrystal energy ---
  p_energy = subparsers.add_parser(
    "energy",
    help="Ground-state energy (auto-selects solver from config)",
  )
  p_energy.add_argument(
    "config", nargs="?", default="config.yaml",
    help="Path to YAML config file (default: config.yaml)",
  )

  # --- jrystal scf ---
  p_scf = subparsers.add_parser(
    "scf", help="Ground-state energy via SCF",
  )
  p_scf.add_argument("config", nargs="?", default="config.yaml")

  # --- jrystal direct-opt ---
  p_do = subparsers.add_parser(
    "direct-opt", help="Ground-state energy via direct optimisation",
  )
  p_do.add_argument("config", nargs="?", default="config.yaml")

  # --- jrystal band ---
  p_band = subparsers.add_parser(
    "band", help="Band structure calculation",
  )
  p_band.add_argument("config", nargs="?", default="config.yaml")

  # --- CLI overrides: --key=value for any config field ---
  args, unknown = parser.parse_known_args()
  overrides = _parse_overrides(unknown)

  config = jr.config.get_config(args.config)
  _apply_overrides(config, overrides)

  # subcommand -> solver.type override
  if args.command == "scf":
    config.solver.type = "scf"
  elif args.command == "direct-opt":
    config.solver.type = "direct_opt"

  # dispatch
  if args.command in ("energy", "scf", "direct-opt"):
    jr.calc.energy(config)
  elif args.command == "band":
    jr.calc.band(config)


def _parse_overrides(args):
  """Parse --key=value or --key value pairs into a dict."""
  overrides = {}
  i = 0
  while i < len(args):
    arg = args[i]
    if arg.startswith("--"):
      if "=" in arg:
        key, value = arg[2:].split("=", 1)
        overrides[key] = value
      elif i + 1 < len(args) and not args[i + 1].startswith("--"):
        overrides[arg[2:]] = args[i + 1]
        i += 1
    i += 1
  return overrides


def _apply_overrides(config, overrides):
  """Apply --key=value overrides to nested config."""
  import yaml
  for key, raw_value in overrides.items():
    value = yaml.safe_load(raw_value)
    parts = key.split(".")
    target = config
    for part in parts[:-1]:
      target = target[part]
    target[parts[-1]] = value
```

**使用效果**:
```bash
jrystal energy                                    # config.yaml, auto solver
jrystal scf si.yaml                               # SCF
jrystal direct-opt si.yaml --solver.epoch=5000    # DO with override
jrystal band si.yaml                              # band
```

**改动范围**: 只改 `main.py`，不动 `calc/__init__.py`。

---

### Step B: 简化 config 字段

**文件**: `jrystal/config.py`

#### B.1: 统一 `system.crystal` + `system.crystal_file_path` → `system.structure`

`default_config` 改动：
```python
"system": {
    "structure": "diamond",     # 替代 crystal + crystal_file_path
    "spin": 0,
    "spin_restricted": True,
},
```

`_LEGACY_FIELD_MAP` 加：
```python
"crystal": ("system", "structure"),
"crystal_file_path": ("system", "structure"),
"crystal_file_path_path": ("system", "structure"),
```

`opt_utils.py` 的 `create_crystal()` 改动：
```python
def create_crystal(config):
    structure = config.system.structure
    if structure is None:
        raise ValueError("system.structure must be set.")

    # Check if it's a built-in name or a file path
    _pkg_path = jr.get_pkg_path()
    builtin_path = f"{_pkg_path}/geometry/{structure}.xyz"
    if os.path.isfile(builtin_path):
        path = builtin_path
    elif os.path.isfile(structure):
        path = structure
    else:
        raise FileNotFoundError(
            f"Cannot find structure '{structure}'. "
            f"Tried built-in ({builtin_path}) and direct path."
        )
    return Crystal.create_from_file(file_path=path, spin=config.system.spin)
```

#### B.2: 统一 `method.use_pseudopotential` + `method.pseudopotential_type` → `method.pseudopotential`

`default_config` 改动：
```python
"method": {
    "xc": "lda_x",
    "pseudopotential": "none",      # "none" | "nc" | "us"
    "pseudopotential_dir": None,    # 替代 pseudopotential_file_dir
},
```

`_LEGACY_FIELD_MAP` 加：
```python
"use_pseudopotential": special handling (True -> "nc", False -> "none"),
"pseudopotential_type": ("method", "pseudopotential"),
"pseudopotential_file_dir": ("method", "pseudopotential_dir"),
```

`backend.py` 的 `get_backend()` 改动：
```python
def get_backend(config):
    pp = config.method.pseudopotential
    if pp in ("none", None, False):
        return AllElectronBackend(config)
    if pp in ("nc", "normcons", "normconserving"):
        return NormConservingBackend(config)
    raise NotImplementedError(...)
```

`opt_utils.py` 的 `create_pseudopotential()` 类似改动。

#### B.3: `_normalize_config` 和 `validate_config` 适配新字段

需要同时支持旧字段（`crystal`, `crystal_file_path`, `use_pseudopotential`）和新字段（`structure`, `pseudopotential`）。migration 逻辑在 `_normalize_config` 中处理。

---

### Step C: 创建示例 input 文件

**新增文件**:
- `examples/input_ae_diamond.yaml` (~7 行)
- `examples/input_nc_diamond.yaml` (~10 行)
- `examples/input_band_diamond.yaml` (~13 行)

这些才是用户应该参考的"最小 input"。当前的 `config.yaml` 改名为 `config_full_reference.yaml` 作为完整参考。

---

### Step D: 更新 `pyproject.toml` entry point

当前:
```toml
[project.scripts]
jrystal = "main:main"
```

不需要改——`main:main` 对应的函数已经在 Step A 中重写了。

---

### Step E: 更新 docs + quickstart

**文件**: `docs/quickstart.md`

改为展示新的 CLI 用法和最小 input 文件：

```markdown
## Quick Start

1. Create a minimal input file `si.yaml`:

\```yaml
system:
  structure: diamond

method:
  xc: lda_x

basis:
  cutoff_energy: 50
  grid_sizes: 24
\```

2. Run:

\```bash
jrystal energy si.yaml
\```

3. For band structure, add a `band` section and run:

\```bash
jrystal band si_band.yaml
\```
```

---

### Step F: 补测试

- `tests/smoke/test_cli.py`: 测 subcommand parsing
- 更新 `tests/smoke/test_config.py`: 测新字段 migration

---

## 六、执行顺序

```
Step B (config 字段简化)   -- 先改底层，这是 foundation
  → Step A (CLI 重写)      -- 依赖 config 字段
  → Step C (示例 input)    -- 依赖新字段格式
  → Step D (pyproject.toml) -- 不需要改
  → Step E (docs)
  → Step F (测试)
```

---

## 七、文件改动清单

| 文件 | 动作 | 说明 |
|------|------|------|
| `jrystal/config.py` | 修改 | `default_config` 字段重命名、migration 适配 |
| `jrystal/calc/opt_utils.py` | 修改 | `create_crystal()` 用 `system.structure`; `create_pseudopotential()` 用 `method.pseudopotential` |
| `jrystal/calc/backend.py` | 修改 | `get_backend()` 用 `method.pseudopotential` |
| `jrystal/calc/runtime.py` | 小改 | 适配新字段名 |
| `jrystal/calc/solver_direct_opt.py` | 小改 | 如果引用了旧字段名 |
| `jrystal/calc/solver_scf.py` | 小改 | 同上 |
| `jrystal/calc/solver_nscf.py` | 小改 | 同上 |
| `main.py` | 重写 | subcommand CLI |
| `config.yaml` | 重命名 | → `config_full_reference.yaml` |
| `examples/input_ae_diamond.yaml` | 新建 | 最小 AE input |
| `examples/input_nc_diamond.yaml` | 新建 | 最小 NC input |
| `examples/input_band_diamond.yaml` | 新建 | 最小 band input |
| `docs/quickstart.md` | 修改 | 新 CLI + 新 input 格式 |
| `tests/smoke/test_cli.py` | 新建 | CLI subcommand 测试 |
| `tests/smoke/test_config.py` | 修改 | 新字段 migration 测试 |
| `tests/reference/run_baseline.py` | 小改 | 适配新字段名 |
