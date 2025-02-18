import tvm
import numpy as np
from tvm import te, auto_scheduler
from tvm import testing
from tvm import auto_tensorize as at
import argparse
import os

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def variance(N, C, H, W, in_dtype, out_dtype):
    A = tvm.te.placeholder([N, C, H, W], dtype=in_dtype, name="A")

    Mean = tvm.te.placeholder([N, C, H, W], dtype=in_dtype, name="Mean")

    Diff = tvm.te.compute(
        [N, C, H, W],
        lambda n, c, h, w: (A[n, c, h, w] - Mean[n, c, h, w]) * (A[n, c, h, w] - Mean[n, c, h, w]),
        name="Diff",
    )

    B = tvm.te.compute(
        [C, N * H * W], lambda i, j: Diff[j // (H * W), i, j % (H * W) // W, j % W], name="B"
    )

    C_ = tvm.te.compute(
        [N * H * W, 1], lambda i, j: tvm.tir.const(1.0 / (N * H * W), in_dtype), name="C"
    )

    rk = tvm.te.reduce_axis([0, N * H * W], name="k")
    D = tvm.te.compute(
        [C, 1], lambda i, j: tvm.te.sum((B[i, rk] * C_[rk, j]).astype(out_dtype), axis=rk), name="D"
    )
    E = tvm.te.compute([C], lambda i: D[i, 0], name="E")
    return [A, B, E]

def run(
    N,
    C,
    H,
    W,
    layer,
    in_dtype,
    out_dtype,
    trials=-1,
):
    log_dir = "variance-%s-%s-layer-%s.json" % (in_dtype, out_dtype, layer)

    target = tvm.target.Target("cuda")
    task = tvm.auto_scheduler.SearchTask(
        func=variance, args=(N, C, H, W, in_dtype, out_dtype), target=target
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

shapes = [
    # resnet-18
    (1, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (1, 64, 56, 56, 64, 64, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    (1, 64, 56, 56, 128, 64, 3, 3, 1, 2, 1, 1, 1),  # conv4   3
    (1, 64, 56, 56, 128, 64, 1, 1, 1, 2, 0, 1, 1),  # conv5   4
    (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 128, 28, 28, 256, 128, 3, 3, 1, 2, 1, 1, 1),  # conv7   6
    (1, 128, 28, 28, 256, 128, 1, 1, 1, 2, 0, 1, 1),  # conv8   7
    (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # conv9   8
    (1, 256, 14, 14, 512, 256, 3, 3, 1, 2, 1, 1, 1),  # conv10  9
    (1, 256, 14, 14, 512, 256, 1, 1, 1, 2, 0, 1, 1),  # conv11  10
    (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # conv12  11
]
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
    python mapping_variance.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_variance.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_variance.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
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
        (N, C, H, W, _, _, _, _, _, _, _, _, _) = shape
        print("\n\nProblem size:")
        print(N, C, H, W)
        layer_name = f"({N}, {C}, {H}, {W})"
        try:
            cost = run(
                N,
                C,
                H,
                W,
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