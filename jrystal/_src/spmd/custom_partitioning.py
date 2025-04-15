from __future__ import annotations

from functools import partial
from typing import Any

import jax
import numpy as np
from jax import tree_util
from jax._src import api_util, config, core, custom_api_util, dispatch
from jax._src import linear_util as lu
from jax._src import mesh as mesh_lib
from jax._src import sharding_impls
from jax._src import xla_bridge as xb
from jax._src.custom_partitioning import (
    _check_for_tracers, _custom_partitioning_abstract_eval,
    _custom_partitioning_impl,
    _custom_partitioning_infer_sharding_from_operands,
    _custom_partitioning_partition,
    _custom_partitioning_propagate_user_sharding, _resolve_kwargs,
    _sharding_callbacks, _ShardingCallbackInfo)
from jax._src.interpreters import batching, mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo

_CUSTOM_PARTITIONING_CALL_NAME = "CustomSPMDPartitioningJrystal"
custom_partitioning_p = core.Primitive("jrystal_custom_partitioning")
custom_partitioning_p.multiple_results = True
dispatch.prim_requires_devices_during_lowering.add(custom_partitioning_p)
custom_partitioning_p.def_abstract_eval(_custom_partitioning_abstract_eval)
custom_partitioning_p.def_impl(_custom_partitioning_impl)


@custom_api_util.register_custom_decorator_type
class jrystal_custom_partitioning:

  def __init__(self, fun, static_argnums=()):
    self.fun = fun
    self.partition = None
    self.static_argnums = static_argnums
    self.propagate_user_sharding = None
    self.infer_sharding_from_operands = None

  __getattr__: Any = custom_api_util.forward_attr

  def def_partition(
    self,
    partition,
    infer_sharding_from_operands,
    propagate_user_sharding=None,
    decode_shardings=True
  ):
    self.partition = partition
    self.propagate_user_sharding = propagate_user_sharding
    self.infer_sharding_from_operands = infer_sharding_from_operands
    self.decode_shardings = decode_shardings
    return partition

  def __call__(self, *args, **kwargs):
    args = _resolve_kwargs(self.fun, args, kwargs)
    debug = api_util.debug_info(
      "custom_partitioning",
      self.fun,
      args,
      kwargs,
      static_argnums=self.static_argnums
    )
    if self.static_argnums:
      static_argnums = set(self.static_argnums)
      args = tuple(x if i in static_argnums else x for i, x in enumerate(args))
      dyn_argnums = [i for i in range(len(args)) if i not in static_argnums]
      f_, dyn_args = api_util.argnums_partial(
          lu.wrap_init(self.fun, debug_info=debug),
          dyn_argnums,
          args,
          require_static_args_hashable=False,
      )
      static_args = [args[i] for i in self.static_argnums]
      _check_for_tracers(static_args)
    else:
      static_args = []
      f_, dyn_args = lu.wrap_init(self.fun, debug_info=debug), args
    args_flat, in_tree = tree_util.tree_flatten(dyn_args)
    flat_fun, out_tree = api_util.flatten_fun_nokwargs(f_, in_tree)
    in_avals = [core.get_aval(x) for x in args_flat]
    mesh = mesh_lib.thread_resources.env.physical_mesh
    with core.extend_axis_env_nd(mesh.shape.items()):
      jaxpr, _, consts, () = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
    assert not len(consts)
    closed_call = core.ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ())

    propagate_user_sharding = None
    infer_sharding_from_operands = None
    sharding_rule = None
    if config.use_shardy_partitioner.value:
      sharding_rule = self.sharding_rule
    else:
      propagate_user_sharding = self.propagate_user_sharding
      infer_sharding_from_operands = self.infer_sharding_from_operands

    out_flat = custom_partitioning_p.bind(
      *consts,
      *args_flat,
      call=closed_call,
      partition=self.partition,
      propagate_user_sharding=propagate_user_sharding,
      infer_sharding_from_operands=infer_sharding_from_operands,
      decode_shardings=self.decode_shardings,
      sharding_rule=sharding_rule,
      in_tree=in_tree,
      out_tree=out_tree(),
      static_args=static_args
    )
    return tree_util.tree_unflatten(out_tree(), out_flat)


def _custom_partitioning_lowering_rule(
  ctx: mlir.LoweringRuleContext,
  *values,
  call,
  in_tree,
  out_tree,
  propagate_user_sharding,
  partition,
  infer_sharding_from_operands,
  decode_shardings,
  sharding_rule,
  static_args
):
  axis_context = ctx.module_context.axis_context
  if (
    isinstance(axis_context, sharding_impls.SPMDAxisContext) and
    set(axis_context.manual_axes) == set(axis_context.mesh.axis_names)
  ):
    return mlir.lower_fun(
      core.jaxpr_as_fun(call), multiple_results=True
    )(ctx, *values)

  mesh = mesh_lib.thread_resources.env.physical_mesh
  if isinstance(axis_context, sharding_impls.ShardingContext):
    devices = axis_context.device_assignment
    if devices is None:
      devices = jax.devices()

    am = axis_context.abstract_mesh
    if am is not None:
      mesh = mesh_lib.Mesh(
        np.array(devices).reshape(am.axis_sizes), am.axis_names
      )
  elif isinstance(axis_context, sharding_impls.SPMDAxisContext):
    devices = axis_context.mesh._flat_devices_tuple
  else:
    devices = None

  if not devices or len(devices) == 1:
    return mlir.lower_fun(
      core.jaxpr_as_fun(call), multiple_results=True
    )(ctx, *values)

  def to_mesh_pspec_sharding(hlo_sharding: xc.HloSharding | None, ndim):
    if hlo_sharding is None:
      return hlo_sharding
    if mesh.empty or not decode_shardings:
      assert devices is not None
      return sharding_impls._op_sharding_to_pos_sharding(hlo_sharding, devices)
    pspec = sharding_impls.parse_flatten_op_sharding(hlo_sharding, mesh)[0]
    pspec = jax.sharding.PartitionSpec(*pspec, *((None,) * (ndim - len(pspec))))
    return jax.sharding.NamedSharding(mesh, pspec)

  sharding_callback_info = _ShardingCallbackInfo(
    propagate_user_sharding,
    partition,
    to_mesh_pspec_sharding,
    in_tree,
    out_tree,
    infer_sharding_from_operands,
    ctx.module_context,
    mesh,
    static_args
  )
  key = str(id(sharding_callback_info))
  _sharding_callbacks[bytes(key, 'utf8')] = sharding_callback_info
  # We need to make sure `sharding_callback_info` is still alive when the SPMD
  # partitioner runs so we keep it alive by attaching it to the executable.
  ctx.module_context.add_keepalive(sharding_callback_info)

  result_types = [mlir.aval_to_ir_type(s) for s in call.out_avals]
  out = hlo.CustomCallOp(
    result_types,
    list(values),
    call_target_name=ir.StringAttr.get(_CUSTOM_PARTITIONING_CALL_NAME),
    has_side_effect=ir.BoolAttr.get(False),
    api_version=mlir.i32_attr(2),
    called_computations=ir.ArrayAttr.get([]),
    backend_config=ir.StringAttr.get(key),
    operand_layouts=None,
    result_layouts=None
  )
  return out.results


mlir.register_lowering(
  custom_partitioning_p, _custom_partitioning_lowering_rule
)

xc.register_custom_call_partitioner(
  _CUSTOM_PARTITIONING_CALL_NAME,
  _custom_partitioning_propagate_user_sharding,
  _custom_partitioning_partition,
  _custom_partitioning_infer_sharding_from_operands,
  can_side_effecting_have_replicated_sharding=True,
)

xb.register_plugin_callbacks(
  partial(
    xc.register_custom_call_partitioner,
    name=_CUSTOM_PARTITIONING_CALL_NAME,
    prop_user_sharding=_custom_partitioning_propagate_user_sharding,
    partition=_custom_partitioning_partition,
    infer_sharding_from_operands=_custom_partitioning_infer_sharding_from_operands,
    can_side_effecting_have_replicated_sharding=True,
  )
)


def _custom_partitioning_batching_rule(
  batched_args, batch_dims, *, call, in_tree, out_tree, **params
):
  # Move batch dimensions to the front
  args_flat = []
  for arg, bdim in zip(batched_args, batch_dims):
    if bdim is not None:
      arg = batching.moveaxis(arg, bdim, 0)
    args_flat.append(arg)

  # Apply the custom partitioning operation
  out_flat = custom_partitioning_p.bind(
    *args_flat, call=call, in_tree=in_tree, out_tree=out_tree, **params
  )

  # Return result with batch dimension at front
  return out_flat, (0,) * len(out_flat)


# Register the batching rule
batching.primitive_batchers[custom_partitioning_p
                           ] = _custom_partitioning_batching_rule
