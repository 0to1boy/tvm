import tvm
import numpy as np
from tvm import te, auto_scheduler
from tvm import testing
from tvm import auto_tensorize as at
import argparse
import os

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def zero_pad2d(inputs, padding=0):
    padding = (
        (padding, padding, padding, padding)
        if isinstance(padding, (int, tvm.tir.IntImm))
        else padding
    )
    assert isinstance(padding, tuple), "type(padding)={}".format(type(padding))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert len(padding) == 4

    padding_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, height, width = inputs.shape
    return tvm.te.compute(
        (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3]),
        lambda b, c, h, w: tvm.te.if_then_else(
            tvm.te.all(
                h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]
            ),
            inputs[b, c, h - padding[0], w - padding[2]],
            padding_zero,
        ),
        name="Padding",
    )

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def capsule_conv2d(
    batch_size,
    in_channel,
    in_h,
    in_w,
    out_channel,
    k_h,
    k_w,
    num_caps,
    stride,
    padding,
    dilation,
    in_dtype,
    out_dtype,
):
    A = tvm.te.placeholder([batch_size, in_channel, in_h, in_w], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([out_channel, in_channel, k_h, k_w, num_caps], dtype=in_dtype, name="B")

    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1

    padded = zero_pad2d(A, padding=padding)
    conv_out_shape = (batch_size, out_channel, out_h, out_w, num_caps)

    rc = tvm.te.reduce_axis((0, in_channel), name="rc")
    rw = tvm.te.reduce_axis((0, k_w), name="rw")
    rh = tvm.te.reduce_axis((0, k_h), name="rh")
    CapsuleConv = tvm.te.compute(
        conv_out_shape,
        lambda b, k, h, w, s: tvm.te.sum(
            (
                padded[b, rc, h * stride + rh * dilation, w * stride + rw * dilation]
                * B[k, rc, rh, rw, s]
            ).astype(out_dtype),
            axis=[rc, rw, rh],
        ),
        name="conv2d_capsule",
    )
    return [A, B, CapsuleConv]
    
def run(
    N,
    C,
    H,
    W,
    K,
    R,
    S,
    num_caps,
    stride,
    padding,
    dilation,
    layer,
    in_dtype,
    out_dtype,
    trials=-1,
):
    log_dir = "capsuleconv2d-%s-%s-layer-%s.json" % (in_dtype, out_dtype, layer)

    target = tvm.target.Target("cuda")
    task = tvm.auto_scheduler.SearchTask(
        func=capsule_conv2d, args=(N, C, H, W, K, R, S, num_caps, stride, padding, dilation, in_dtype, out_dtype), target=target
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
    # resnet-18
    # (batch, C, H, W, K, _, R, S, num_caps, stride, padding, dilation, groups)
    (1, 3, 224, 224, 64, 3, 7, 7, 8, 2, 3, 1, 1),  # conv1  0
    (1, 64, 56, 56, 64, 64, 3, 3, 8, 1, 1, 1, 1),  # conv2   1
    (1, 64, 56, 56, 64, 64, 1, 1, 8, 1, 0, 1, 1),  # conv3   2
    (1, 64, 56, 56, 128, 64, 3, 3, 8, 2, 1, 1, 1),  # conv4   3
    (1, 64, 56, 56, 128, 64, 1, 1, 8, 2, 0, 1, 1),  # conv5   4
    (1, 128, 28, 28, 128, 128, 3, 3, 8, 1, 1, 1, 1),  # conv6   5
    (1, 128, 28, 28, 256, 128, 3, 3, 8, 2, 1, 1, 1),  # conv7   6
    (1, 128, 28, 28, 256, 128, 1, 1, 8, 2, 0, 1, 1),  # conv8   7
    (1, 256, 14, 14, 256, 256, 3, 3, 8, 1, 1, 1, 1),  # conv9   8
    (1, 256, 14, 14, 512, 256, 3, 3, 8, 2, 1, 1, 1),  # conv10  9
    (1, 256, 14, 14, 512, 256, 1, 1, 8, 2, 0, 1, 1),  # conv11  10
    (1, 512, 7, 7, 512, 512, 3, 3, 8, 1, 1, 1, 1),  # conv12  11
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
    python mapping_capsuleconv2d.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_capsuleconv2d.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_capsuleconv2d.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
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
            (_, C, H, W, K, _, R, S, num_caps, stride, padding, dilation, _) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding, dilation)
            layer_name = f"({N},{C},{H},{W},{K},{R},{S},{num_caps},{stride},{padding},{dilation})"
            try:
                cost = run(
                    N,
                    C,
                    H,
                    W,
                    K,
                    R,
                    S,
                    num_caps,
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