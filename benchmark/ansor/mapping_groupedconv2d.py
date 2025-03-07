import tvm
import numpy as np
from tvm import te, auto_scheduler
from tvm import testing
from tvm import auto_tensorize as at
import argparse
import os

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def grouped_conv2d(N, C, H, W, K, R, S, stride, padding, groups, in_dtype, out_dtype):
    assert K % groups == 0
    assert C % groups == 0
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([K, C // groups, R, S], dtype=in_dtype, name="B")

    Pad = tvm.te.compute(
        [N, C, pH, pW],
        lambda n, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
            A[n, c, h - padding, w - padding],
            tvm.tir.const(0.0, A.dtype),
        ),
        name="Pad",
    )

    A_reshaped = tvm.te.compute(
        [N, groups, C // groups, pH, pW],
        lambda n, c_o, c_i, r, s: Pad[n, c_o * (C // groups) + c_i, r, s],
    )

    B_reshaped = tvm.te.compute(
        [groups, K // groups, C // groups, R, S],
        lambda k_o, k_i, c, r, s: B[k_o * (K // groups) + k_i, c, r, s],
    )

    rc = tvm.te.reduce_axis([0, C // groups], name="rc")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    P = (pH - R) // stride + 1
    Q = (pW - S) // stride + 1
    Conv = tvm.te.compute(
        [N, groups, K // groups, P, Q],
        lambda n, k_o, k_i, p, q: tvm.te.sum(
            (
                A_reshaped[n, k_o, rc, p * stride + rr, q * stride + rs]
                * B_reshaped[k_o, k_i, rc, rr, rs]
            ).astype(out_dtype),
            axis=[rc, rr, rs],
        ),
        name="Conv",
    )

    Conv_reshaped = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q: Conv[n, k // (K // groups), k % (K // groups), p, q],
        name="Reshaped",
    )

    return [A, B, Conv_reshaped]

def run(
    N,
    C,
    H,
    W,
    K,
    R,
    S,
    stride,
    padding,
    groups,
    layer,
    in_dtype,
    out_dtype,
    trials=-1,
):
    log_dir = "grouped_conv2d-%s-%s-layer-%s.json" % (in_dtype, out_dtype, layer)

    target = tvm.target.Target("cuda")
    task = tvm.auto_scheduler.SearchTask(
        func=grouped_conv2d, args=(N, C, H, W, K, R, S, stride, padding, groups, in_dtype, out_dtype), target=target
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

_ = -1
shapes_b1 = [
    #   (N,   C,   H,   W,   K, R,  S,  _, stride, padding, dilation, groups, _)
    (1, 3, 224, 224, 24, 3, 3, _, 2, 1, 1, 3, _),
    # ShuffleNetUnitB + ShuffleNetUnitA x 3
    # cudnn doesn't support this one
    # (1,  24,  56,  56,  54,   1,  1,  _,      1,       0,        1,      3, _),
    # (1,  54,  56,  56,  54,   3,  3,  _,      2,       1,        1,     54, _),
    (1, 54, 28, 28, 216, 1, 1, _, 1, 0, 1, 3, _),
    (1, 240, 28, 28, 60, 1, 1, _, 1, 0, 1, 3, _),
    # (1,  60,  28,  28,  60,   3,  3,  _,      1,       1,        1,     60, _),
    (1, 60, 28, 28, 240, 1, 1, _, 1, 0, 1, 3, _),
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
    python mapping_groupedconv2d.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_groupedconv2d.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_groupedconv2d.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=1)
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
    parser.add_argument("--begin", type=int, choices=list(range(len(shapes_b1))), default=0)
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes_b1) + 1)), default=len(shapes_b1)
    )
    parser.add_argument("--trials", type=int, default=-1)

    args = parser.parse_args()
    batches = [args.batch]
    beg = args.begin
    num = args.num
    assert (
        args.in_dtype,
        args.out_dtype,
    ) in supported_dtypes, (
        f"The desired dtype pair {(args.in_dtype, args.out_dtype)} is not supported by Tensor Core."
    )
    for batch in batches:
        costs = []
        for i, shape in enumerate(shapes_b1[beg : beg + num]):
            (_, C, H, W, K, R, S, _, stride, padding, _, groups, _) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding, groups)
            layer_name = f"({N},{C},{H},{W},{K},{R},{S},{stride},{padding},{groups})"
            try:
                cost = run(
                    N,
                    C,
                    H,
                    W,
                    K,
                    R,
                    S,
                    stride,
                    padding,
                    groups,
                    layer_name,
                    args.in_dtype,
                    args.out_dtype,
                    trials=args.trials,
                )
                costs.append(cost)
            except Exception as e:
                print("Fail to run\n", str(e))
                costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)