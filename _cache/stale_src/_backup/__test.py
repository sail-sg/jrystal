import numpy as np
import jax
import jax.numpy as jnp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import multiprocessing as mp


# 设置JAX不使用所有GPU内存，留出空间给其他进程
jax.config.update('jax_platform_name', 'gpu')  # 明确使用GPU
# jax.config.update('jax_default_matmul_precision', 'float32')
mp.set_start_method("spawn", force=True)


def numpy_heavy_computation(data):
  result = np.linalg.svd(data)
  return result


@jax.jit
def jax_heavy_computation(data):
  return jnp.linalg.svd(data)


def main():
  cpu_data = [np.random.rand(1000, 1000) for _ in range(8)]
  gpu_data = jnp.array(np.random.rand(5000, 5000))

  with ProcessPoolExecutor(max_workers=cpu_count()//2) as executor:
    cpu_results = list(executor.map(numpy_heavy_computation, cpu_data))

  gpu_result = jax_heavy_computation(gpu_data)

  final_result = (cpu_results, gpu_result)
  return final_result

if __name__ == '__main__':
    result = main()