# Jrystal Refactoring Plan Review

我读完了 `codex.md`（总体规划）和 `codex1.md`（第一步详细计划），也通读了项目核心代码。下面分三块：**计划中不合理/有风险的地方**、**可以改进的建议**、**需要你确认的问题**。

---

## 一、计划中不合理或有风险的地方

### 1. `calc/__init__.py` 当前会直接 break

现在 `calc/__init__.py:19` 导入的是：
```python
from .calc_ground_state_energy_normcons_cg import calc as energy_normcons
```
这个文件 `import jaxopt`（第 23 行），但 `jaxopt` 不在 `pyproject.toml` 的依赖里。也就是说**当前 `import jrystal.calc` 在没装 jaxopt 的环境下就会直接报错**。

计划里把"去掉 jaxopt"放在 WP6（第 7 步），但 WP1 的验收标准就要求 `import jrystal.calc` 可用。**这两件事有硬依赖，WP1 开始之前就必须先处理 jaxopt 问题**，否则 WP1 的验收标准永远过不了。

**建议**：把"去掉 jaxopt 路线"提到最前面做，或者至少在 WP1 里一并处理。这只是改一行 import + 删/archive 一个文件的事。

### 2. `create_grids()` 丢弃 k_weights 的问题比文档描述的更严重

`k_vectors()` 已经返回 `(k_mesh, k_weights)`，但 `opt_utils.py:149` 的 `create_grids()` 写的是：
```python
kpts = k_vectors(crystal.cell_vectors, k_grid_sizes)
```
这里有两个问题：
- `k_vectors()` 现在 **必须** 传 `scaled_positions` 和 `charges`（symmetry_reduction 默认为 True），所以这行代码在当前 main 分支上**应该已经报错了**。
- 即使修好了调用签名，它也只返回了 `kpts` 而丢弃了 `weights`。

这说明 `create_grids()` 在当前代码里可能已经不是被 normcons workflow 实际使用的路径了（normcons 的 `calc` 函数直接调用 `create_grids` 但看起来没有用 weights）。**需要确认当前的 normcons workflow 到底有没有在用 k-point weights**，如果没有，那结果的物理正确性可能本身就存疑。

### 3. `GroundStateEnergyOutput` 定义了但从未被返回

`calc_ground_state_energy_normcons.py` 和 `calc_ground_state_energy_all_electrons.py` 都定义了 `GroundStateEnergyOutput` dataclass，但两个文件最后都是 `return density`。这个问题计划里提到了（WP5），但我觉得优先级应该更高——**如果 band workflow 依赖 ground-state 的 result 对象（`calc_band_structure_normcons.py:32` 导入了 `GroundStateEnergyOutput`），那在 result 对象修好之前，band workflow 的 pipeline 其实是断的**。

### 4. SCF 代码是独立脚本，不是可集成的模块

`jrystal/scf/scf.py` 和 `jrystal/scf/scf_spmd.py` 都是硬编码路径、硬编码参数的脚本文件（顶层有 `jax.config.update`，路径写死为 `/home/aiops/litb/...`）。计划里提到 SCF 作为 workflow solver，但没有明确说明这些脚本如何被重构成可调用的模块。**如果 Phase 1 的目标包含 SCF 作为正式 solver，那 WP 里应该有一个专门的步骤来处理 scf 模块的模块化**。

### 5. 计划的 8 个 WP 之间的依赖关系不够清晰

计划说"严格按顺序"，但实际上有些 WP 可以并行，有些有硬前置依赖：
- WP1（package surface）**被 WP6（去 jaxopt）阻塞**（如上所述）
- WP4（RuntimeContext）依赖 WP3（KSampling），这个顺序是对的
- WP5（result dataclass）理论上可以和 WP3/WP4 并行
- WP7（测试）其实应该从 WP1 开始就逐步搭建，而不是等到第 8 步

**建议**：画一个简单的依赖图，哪些可以并行，哪些有硬前置。

---

## 二、可以改进的建议

### 1. 先做一个 "ground truth" 基线

在动任何代码之前，建议先用当前代码跑一个 canonical benchmark（比如 Si diamond），把能量值记录下来。这样重构过程中可以随时验证数值没有 regress。这比 WP7 搭测试骨架优先级更高，因为它是零成本的——只需要跑一次、记一个数。

### 2. `ElectronicBackend` 的时机

`codex.md` 提出了 `ElectronicBackend` 接口（AE/NC/USPP 共用 workflow），这是一个很好的设计方向。但 `codex1.md`（第一步）里完全没有提这个。如果第一步不引入 `ElectronicBackend`，那后面引入时就会面临第二次大规模重构。

**建议**：第一步不需要完整实现 `ElectronicBackend`，但可以先定义一个最小 protocol/interface，让 NC 的代码先适配进去。这样第二步加 USPP 时就不需要再动 workflow 壳子了。

### 3. Config 迁移策略需要更具体

WP2 说要"明确分块语义，但先兼容旧 flat yaml"。但现在 `config.yaml` 是完全 flat 的，新 schema 要分成 `system/method/basis/ksampling/solver/execution/io`。怎么兼容？两种策略差别很大：
- **策略 A**：load 时自动把 flat key 映射到 nested structure（需要写一个 migration 函数）
- **策略 B**：同时支持 flat 和 nested 两种格式（更复杂但对用户更友好）

建议明确选一个。

### 4. `ml_collections.ConfigDict` 是否继续用

现在 config 基于 `ml_collections.ConfigDict`。如果要加 `schema_version`、分块语义、unknown field 校验，`ConfigDict` 其实不太适合——它本质上就是一个 dict wrapper，没有 schema validation 能力。可以考虑：
- 切到 `dataclass` + `dacite` 或 `pydantic`
- 或者继续用 `ConfigDict` 但手写 validation

这个选择会影响 WP2 的工作量，需要提前决定。

### 5. `_src` 里的 proxy module 层

当前 `jrystal/energy.py`、`jrystal/pw.py`、`jrystal/hamiltonian.py` 等都是从 `_src` re-export 的 proxy module。但 `calc` 层的代码直接 import `_src`（如 `from .._src import energy, pw`）。这意味着这些 proxy module 在 calc 层完全没有被使用。

如果要收紧 package surface，需要决定：
- `calc` 层是通过 proxy module 访问 `_src`，还是直接访问 `_src`？
- proxy module 存在的意义是什么？是给外部用户用的 public API 吗？

### 6. 测试不应该等到 WP7

计划把测试放到第 8 步。但重构最大的风险就是 regression。**建议从 WP1 开始，每完成一个 WP 就补对应的 smoke test**。比如：
- WP1 做完 → 补 import smoke test
- WP2 做完 → 补 config load test
- WP3 做完 → 补 KSampling 构造 test

这样每步都有安全网。

---

## 三、需要你确认的问题

### Q1: 当前 normcons workflow 的 k-point weights 状态

`create_grids()` 调用 `k_vectors()` 但不传 `scaled_positions`/`charges`，而且丢弃 weights。那当前 `calc_ground_state_energy_normcons.py` 的 ground-state 计算到底有没有正确使用 k-point weights？还是说它目前只跑 Gamma-only 或者用 uniform weights 凑合？

**这个问题决定了 WP3（KSampling）的实际紧迫程度**——如果当前结果本来就没用 weights，那 K-point weights 的正确接入本身就是一个 physics fix，不只是 contract 收敛。

### Q2: SCF 在第一步的定位

`codex.md` Phase 2 才做 SCF。但 `jrystal/scf/` 下已经有了 SCF + DIIS + LOBPCG 的实现（虽然是脚本形态）。第一步是：
- (a) 只收敛 DirectOpt workflow，SCF 不动？
- (b) 顺便把 SCF 也模块化进来？

如果选 (a)，`jrystal/scf/` 这个目录在第一步应该怎么处理——留着还是先 archive？

### Q3: `pseudopotential_type` 字段

`config.yaml` 里有 `pseudopotential_type: "nc"`，但 `default_config`（`config.py:61`）里没有这个字段。`opt_utils.py:121` 又引用了 `config.pseudopotential_type`。这说明默认 config 和实际使用的 config schema 已经不一致了。第一步修 config 时是否要补齐这些缺失字段？

### Q4: `chex.dataclass` vs `dataclasses.dataclass`

`Crystal` 用的是 `chex.dataclass`（支持 JAX tree），`GroundStateEnergyOutput` 用的是标准 `dataclasses.dataclass`。新引入的 `RuntimeContext`、`KSampling`、`GroundStateResult` 等要用哪个？如果它们需要被 `jax.jit` 边界传递，就必须是 pytree-compatible 的。

### Q5: `jrystal.calc` vs `jrystal.workflows` 命名

`codex1.md` 在 WP3/WP4 里提到了两个可选路径：
- `jrystal/calc/types.py` 或 `jrystal/workflows/types.py`
- `jrystal/calc/runtime.py` 或 `jrystal/workflows/runtime.py`

你更倾向于哪个？如果统一用 `workflows`，那 `calc` 这个 module 最终是要废弃还是保留？

### Q6: 多少个 calc 文件要保留

当前 `calc/` 下有这些实际 workflow 文件：
- `calc_ground_state_energy_all_electrons.py`
- `calc_ground_state_energy_normcons.py`
- `calc_ground_state_energy_normcons_cg.py`（jaxopt，要删）
- `calc_ground_state_energy_normcons_psgd.py`
- `calc_ground_state_energy_ultrasoft.py`
- `calc_ground_state_energy_hartree_fock.py`
- `calc_band_structure_all_electrons.py`
- `calc_band_structure_normcons.py`
- `calc_band_structure_ultrasoft.py`
- `geo_opt_all_electron.py`
- `preconditioned_sgd.py`

加上 `_backup/` 下的一堆。第一步打算保留哪些、archive 哪些、删除哪些？这直接影响 WP1 的工作量。

### Q7: energy 函数里的 k-point weights

我看了 `_src/energy.py` 的 `kinetic`、`hartree` 等函数——这些函数的签名里似乎没有显式的 `k_weights` 参数。那 k-point weights 目前是怎么被消费的？是在 occupation 层面乘进去的，还是在 energy 汇总时乘的？这关系到 `KSampling` 对象应该在 pipeline 的哪一层被消费。

### Q8: 这次重构是在 `scf/refactor` 分支上做吗？

当前 branch 是 `scf/refactor`，但这次重构的范围远大于 SCF。是新开一个 branch（如 `refactor/step-1`），还是继续在这个 branch 上做？

---

## 四、建议的调整后执行顺序

基于以上分析，我建议的实际执行顺序：

```
0. 跑一个 baseline benchmark，记录当前 Si diamond 能量值
1. 去掉 jaxopt 路线 (原 WP6，前置依赖)
2. 收紧 package surface (原 WP1) + 同时补 import smoke test
3. 收敛 config contract (原 WP2) + 补 config smoke test
4. 引入 KSampling (原 WP3) + 补 KSampling test
5. 引入 RuntimeContext (原 WP4) + 定义最小 ElectronicBackend protocol
6. 统一 result dataclass (原 WP5) + 补 workflow smoke test
7. 文档与 examples 对齐 (原 WP8)
```

主要变化：
- jaxopt 提前到第 1 步（解除 import 阻塞）
- 测试不再是独立的一步，而是分散到每个 WP 中
- 在 WP4 时顺便定义 `ElectronicBackend` 的最小 protocol
- baseline benchmark 在一切之前
