import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from memory_profiler import memory_usage
import time

# 生成模拟数据
n_samples = 40000
n_features = 3
y = np.random.rand(n_samples, n_features)


# 默认非支配排序
def test_default_non_dominated_sort():
    sorting = NonDominatedSorting()
    sorting.do(y)


# 高效非支配排序
def test_efficient_non_dominated_sort():
    sorting = NonDominatedSorting(method="efficient_non_dominated_sort")
    sorting.do(y)


def run_test(func, name):
    print(f"\nTesting {name}...")

    start_time = time.time()

    # memory_usage 会返回峰值内存
    max_mem = memory_usage(
        (func, (), {}),
        max_usage=True,
        interval=0.1
    )

    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Max memory usage: {max_mem:.2f} MB")


if __name__ == "__main__":

    run_test(test_efficient_non_dominated_sort, "efficient NonDominatedSorting")

    run_test(test_default_non_dominated_sort, "default NonDominatedSorting")