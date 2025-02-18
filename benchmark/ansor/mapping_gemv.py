import tvm
import numpy as np
from tvm import te, auto_scheduler
from tvm import testing
from tvm import auto_tensorize as at
import argparse
import os

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def gemv(M, K, in_dtype, out_dtype):
    A = tvm.te.placeholder([M, K], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([K, 1], dtype=in_dtype, name="B")

    rk = tvm.te.reduce_axis([0, K], name="k")
    C = tvm.te.compute(
        [M, 1], lambda i, j: tvm.te.sum((A[i, rk] * B[rk, j]).astype(out_dtype), axis=rk), name="C"
    )
    D = tvm.te.compute([M], lambda i: C[i, 0], name="D")
    return [A, B, D]

def run(
    M,
    K,
    layer,
    in_dtype,
    out_dtype,
    trials=-1,
):
    log_dir = "gemv-%s-%s-layer-%s.json" % (in_dtype, out_dtype, layer)

    target = tvm.target.Target("cuda")
    task = tvm.auto_scheduler.SearchTask(func=gemv, args=(M, K, in_dtype, out_dtype), target=target)

    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials,
        measure_callbacks=[auto_scheduler.RecordToFile(log_dir)],
        verbose=2,
    )

    # Run auto-tuning (search)
    task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_dir)

    measure_opt = at.MeasureOptions(target="cuda", timeout=10, number=200, min_repeat_ms=500)
    cost = at.evaluate_schedule(sch, args, measure_opt)
    print("Cost is %f ms" % cost)
    return cost

shapes = [(512, 16), (1024, 256), (256, 1024), (512, 256), (1024, 1024)]
supported_dtypes = set(
    [
        ("float16", "float16"),
        ("float16", "float32"),
        ("bfloat16", "float32"),
        ("float32", "float32"),
        ("float64", "float64"),
        ("int4", "int32"),
        ("int8", "int32"),
    ]
)

example_text = """
 example:
    python mapping_gemv.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_gemv.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_gemv.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--in_dtype",
        type=str,
        choices=["float16", "float32", "float64", "bfloat16", "int4", "int8"],
        default="float16",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        choices=["float16", "float32", "float64", "int32"],
        default="float16",
    )
    parser.add_argument("--begin", type=int, choices=list(range(len(shapes))), default=0)
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )
    parser.add_argument("--trials", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_perf_model", action="store_true")
    parser.add_argument("--perf_model_ratio", type=float, default=0.6)

    args = parser.parse_args()
    beg = args.begin
    num = args.num
    assert (
        args.in_dtype,
        args.out_dtype,
    ) in supported_dtypes, (
        f"The desired dtype pair {(args.in_dtype, args.out_dtype)} is not supported by Tensor Core."
    )
    costs = []
    for i, shape in enumerate(shapes[beg : beg + num]):
        (M, K) = shape
        print("\n\nProblem size:")
        print(M, K)
        layer_name = f"({M}, {K})"
        try:
            cost = run(
                M,
                K,
                layer_name,
                args.in_dtype,
                args.out_dtype,
                trials=args.trials,
            )
            costs.append(cost)
        except Exception as e:
            print("Fail to run\n", str(e))
            costs.append(float("inf"))
    for cost in costs:
        print(cost)
