import tvm
import numpy as np
from tvm import te, auto_scheduler
from tvm import testing
from tvm import auto_tensorize as at
import argparse
import os

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def scan(M, N, K, in_dtype, out_dtype):
    A = tvm.te.placeholder([M, K], dtype=in_dtype, name="A")
    B = tvm.te.compute(
        [K, N],
        lambda i, j: tvm.tir.if_then_else(
            i > j, tvm.tir.const(0.0, in_dtype), tvm.tir.const(1.0, "float16")
        ),
        name="B",
    )

    rk = tvm.te.reduce_axis([0, K], name="k")
    C = tvm.te.compute(
        [M, N], lambda i, j: tvm.te.sum((A[i, rk] * B[rk, j]).astype(out_dtype), axis=rk), name="C"
    )
    return [A, B, C]

def run(
    M,
    N,
    K,
    layer,
    in_dtype,
    out_dtype,
    trials=-1,
):
    log_dir = "scan-%s-%s-layer-%s.json" % (in_dtype, out_dtype, layer)

    target = tvm.target.Target("cuda")
    task = tvm.auto_scheduler.SearchTask(
        func=scan, args=(M, N, K, in_dtype, out_dtype), target=target
    )

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

shapes = [(16, 512, 128), (1024, 16, 256), (256, 1024, 256), (512, 256, 16), (1024, 1024, 1024)]

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
    python mapping_scan.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_scan.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_scan.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
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
        (M, N, K) = shape
        print("\n\nProblem size:")
        print(M, N, K)
        layer_name = f"({M}, {N}, {K})"
        try:
            cost = run(
                M,
                N,
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
