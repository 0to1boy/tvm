import tvm
import numpy as np
from tvm import te, auto_scheduler
from tvm import testing
from tvm import auto_tensorize as at
import argparse
import os

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def batch_conv2d(N, C, H, W, K, R, S, stride, padding, in_dtype, out_dtype):
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([N, K, C, R, S], dtype=in_dtype, name="B")

    Pad = tvm.te.compute(
        [N, C, pH, pW],
        lambda n, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
            A[n, c, h - padding, w - padding],
            tvm.tir.const(0.0, A.dtype),
        ),
        name="Pad",
    )

    rc = tvm.te.reduce_axis([0, C], name="rc")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    P = (pH - R) // stride + 1
    Q = (pW - S) // stride + 1
    Conv = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q: tvm.te.sum(
            (Pad[n, rc, p * stride + rr, q * stride + rs] * B[n, k, rc, rr, rs]).astype(out_dtype),
            axis=[rc, rr, rs],
        ),
        name="Conv",
    )
    return [A, B, Conv]

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
    layer,
    in_dtype,
    out_dtype,
    trials=-1,
):
    log_dir = "batch-conv2d-%s-%s-layer-%s.json" % (in_dtype, out_dtype, layer)

    target = tvm.target.Target("cuda")
    task = tvm.auto_scheduler.SearchTask(
        func=batch_conv2d, args=(N, C, H, W, K, R, S, stride, padding, in_dtype, out_dtype), target=target
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

shapes_b1 = [
    (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),
    (1, 64, 56, 56, 64, 64, 1, 1, 1, 1, 0, 1, 1),
    (1, 64, 56, 56, 128, 64, 3, 3, 1, 2, 1, 1, 1),
    (1, 64, 56, 56, 128, 64, 1, 1, 1, 2, 0, 1, 1),
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
    python mapping_batchconv2d.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_batchconv2d.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_batchconv2d.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
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
            (_, C, H, W, K, _, R, S, _, stride, padding, _, _) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding)
            layer_name = f"({N},{C},{H},{W},{K},{R},{S},{stride},{padding})"
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