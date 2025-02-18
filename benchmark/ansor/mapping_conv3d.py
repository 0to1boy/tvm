import tvm
import numpy as np
from tvm import te, auto_scheduler
from tvm import testing
from tvm import auto_tensorize as at
import argparse
import os

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def conv3d(
    N, C, D, H, W, K, KD, R, S, stride_d, stride, padding_d, padding, dilation, in_dtype, out_dtype
):
    pD = D + 2 * padding_d
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, D, H, W], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([K, C, KD, R, S], dtype=in_dtype, name="B")

    Pad = tvm.te.compute(
        [N, C, pD, pH, pW],
        lambda n, c, d, h, w: tvm.tir.if_then_else(
            tvm.tir.all(
                d >= padding_d,
                d - padding_d < D,
                h >= padding,
                h - padding < H,
                w >= padding,
                w - padding < W,
            ),
            A[n, c, d - padding_d, h - padding, w - padding],
            tvm.tir.const(0.0, A.dtype),
        ),
        name="Pad",
    )

    rc = tvm.te.reduce_axis([0, C], name="rc")
    rd = tvm.te.reduce_axis([0, KD], name="rd")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    outD = (pD - dilation * (KD - 1) - 1) // stride_d + 1
    P = (pH - dilation * (R - 1) - 1) // stride + 1
    Q = (pW - dilation * (S - 1) - 1) // stride + 1

    Conv = tvm.te.compute(
        [N, K, outD, P, Q],
        lambda n, k, d, p, q: tvm.te.sum(
            (
                Pad[
                    n, rc, d * stride_d + rd, p * stride + rr * dilation, q * stride + rs * dilation
                ]
                * B[k, rc, rd, rr, rs]
            ).astype(out_dtype),
            axis=[rc, rd, rr, rs],
        ),
        name="Conv",
    )
    return [A, B, Conv]

def run(
    N,
    C,
    D,
    H,
    W,
    K,
    KD,
    R,
    S,
    stride_d,
    stride,
    padding_d,
    padding,
    dilation,
    layer,
    in_dtype,
    out_dtype,
    trials=-1,
):
    log_dir = "conv3d-%s-%s-layer-%s.json" % (in_dtype, out_dtype, layer)

    target = tvm.target.Target("cuda")
    task = tvm.auto_scheduler.SearchTask(
        func=conv3d, 
        args=(
            N,
            C,
            D,
            H,
            W,
            K,
            KD,
            R,
            S,
            stride_d,
            stride,
            padding_d,
            padding,
            dilation,
            in_dtype,
            out_dtype,
        ), 
        target=target
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


_ = None
L = 8


#  (  N,   C,     L,   H,   W,   K,   D,   R,   S, stride_d, stride, padding_d, padding, dilation)
shapes_b1 = [
    (_, 3, L, 112, 112, 64, 1, 3, 3, 3, 7, 1, 3, 1),  # stem
    (_, 64, L, 56, 56, 64, 3, 3, 3, 1, 1, 1, 1, 1),  # layer1 x 4
    (_, 64, L, 56, 56, 128, 1, 1, 1, 2, 2, 0, 0, 1),  # layer2 downsample
    (_, 64, L, 56, 56, 128, 3, 3, 3, 2, 2, 1, 1, 1),  # layer2
    (_, 128, L // 2, 28, 28, 128, 3, 3, 3, 1, 1, 1, 1, 1),  # layer2 x 3
    (_, 128, L // 2, 28, 28, 256, 1, 1, 1, 2, 2, 0, 0, 1),  # layer3 downsample
    (_, 128, L // 2, 28, 28, 256, 3, 3, 3, 2, 2, 1, 1, 1),  # layer3
    (_, 256, L // 4, 14, 14, 256, 3, 3, 3, 1, 1, 1, 1, 1),  # layer3 x 3
    (_, 256, L // 4, 14, 14, 512, 1, 1, 1, 2, 2, 0, 0, 1),  # layer4 downsample
    (_, 256, L // 4, 14, 14, 512, 3, 3, 3, 2, 2, 1, 1, 1),  # layer4
    (_, 256, L // 8, 7, 7, 512, 3, 3, 3, 1, 1, 1, 1, 1),  # layer4 x 3
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
    python mapping_conv3d.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_conv3d.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_conv3d.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
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
            (_, C, D, H, W, K, KD, R, S, stride_d, stride, padding_d, padding, dilation) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, D, H, W, K, KD, R, S, stride_d, stride, padding_d, padding, dilation)
            layer_name = f"({N},{C},{D},{H},{W},{K},{KD},{R},{S},{stride_d},{stride},{padding_d},{padding},{dilation})"
            try:
                cost = run(
                    N,
                    C,
                    D,
                    H,
                    W,
                    K,
                    KD,
                    R,
                    S,
                    stride_d,
                    stride,
                    padding_d,
                    padding,
                    dilation,
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