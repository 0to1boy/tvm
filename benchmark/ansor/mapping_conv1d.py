import tvm
import numpy as np
from tvm import te, auto_scheduler
from tvm import testing
from tvm import auto_tensorize as at
import argparse
import os

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def zero_pad1d(inputs, padding=0):
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    assert isinstance(padding, tuple), "type(padding)={}".format(type(padding))
    assert len(padding) == 2

    padding_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, in_len = inputs.shape
    return tvm.te.compute(
        (batch_size, in_channel, in_len + padding[0] + padding[1]),
        lambda b, c, l: tvm.te.if_then_else(
            tvm.te.all(l >= padding[0], l < in_len + padding[0]),
            inputs[b, c, l - padding[0]],
            padding_zero,
        ),
    )

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def conv1d(N, C, L, K, KL, stride, padding, dilation, in_dtype, out_dtype):
    A = tvm.te.placeholder([N, C, L], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([K, C, KL], dtype=out_dtype, name="B")

    out_len = (L + 2 * padding - dilation * (KL - 1) - 1) // stride + 1

    rc = tvm.te.reduce_axis((0, C), name="rc")
    rl = tvm.te.reduce_axis((0, KL), name="rl")

    padded = zero_pad1d(A, padding=padding)
    conved = tvm.te.compute(
        (N, K, out_len),
        lambda b, k, l: tvm.te.sum(
            (padded[b, rc, l * stride + rl * dilation] * B[k, rc, rl]).astype(out_dtype),
            axis=[rc, rl],
        ),
    )
    return [A, B, conved]

def run(
    N,
    C,
    L,
    K,
    KL,
    stride,
    padding,
    dilation,
    layer,
    in_dtype,
    out_dtype,
    trials=-1,
):
    log_dir = "conv1d-%s-%s-layer-%s.json" % (in_dtype, out_dtype, layer)

    target = tvm.target.Target("cuda")
    task = tvm.auto_scheduler.SearchTask(func=conv1d, args=(N, C, L, K, KL, stride, padding, dilation, in_dtype, out_dtype), target=target)

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
    # byte_net_shapes
    # (   C,   L,   K,  KL, stride,padding, dilation)
    (512, 892, 512, 3, 1, 2, 1),
    (512, 892, 1024, 1, 1, 0, 1),
    (1024, 892, 512, 1, 1, 0, 1),
    (512, 892, 512, 3, 1, 4, 2),
    (512, 892, 512, 3, 1, 8, 4),
    (512, 892, 512, 3, 1, 16, 8),
    (512, 892, 512, 3, 1, 32, 16),
    (1024, 892, 250, 1, 1, 0, 1),
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
    python mapping_conv1d.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_conv1d.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_conv1d.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
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
            (C, L, K, KL, stride, padding, dilation) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, L, K, KL, stride, padding, dilation)
            layer_name = f"({N},{C},{L},{K},{KL},{stride},{padding},{dilation})"
            try:
                cost = run(
                    N,
                    C,
                    L,
                    K,
                    KL,
                    stride,
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