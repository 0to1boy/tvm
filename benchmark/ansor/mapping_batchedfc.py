import tvm
import numpy as np
from tvm import te, auto_scheduler
from tvm import testing
from tvm import auto_tensorize as at
import argparse
import os

@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def batched_fc(N, I, O, groups, in_dtype, out_dtype):
    channel_per_group = I // groups
    out_channel_per_group = O // groups

    A = tvm.te.placeholder([N, I], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([O, channel_per_group], dtype=in_dtype, name="B")

    A_reshaped = tvm.te.compute(
        [N, groups, channel_per_group], lambda n, c_o, c_i: A[n, c_o * channel_per_group + c_i]
    )

    B_reshaped = tvm.te.compute(
        [groups, out_channel_per_group, channel_per_group],
        lambda k_o, k_i, c: B[k_o * out_channel_per_group + k_i, c],
    )

    rc = tvm.te.reduce_axis([0, channel_per_group], name="rc")

    BFC = tvm.te.compute(
        [N, groups, out_channel_per_group],
        lambda n, k_o, k_i: tvm.te.sum(
            (A_reshaped[n, k_o, rc] * B_reshaped[k_o, k_i, rc]).astype(out_dtype),
            axis=[
                rc,
            ],
        ),
        name="BatchedFC",
    )

    return [A, B, BFC]

def run(
    N,
    I,
    O,
    groups,
    layer,
    in_dtype,
    out_dtype,
    trials=-1,
):
    log_dir = "batched-fc-%s-%s-layer-%s.json" % (in_dtype, out_dtype, layer)

    target = tvm.target.Target("cuda")
    task = tvm.auto_scheduler.SearchTask(
        func=batched_fc, args=(N, I, O, groups, in_dtype, out_dtype), target=target
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
    # https://github.com/megvii-model/WeightNet/blob/master/shufflenet_v2.py
    # in_channels, out_channels, groups (ksize, stride, padding = 1, 1, 0)
    # shuffle_v2_cfg
    (24, 216, 24),
    (48, 576, 48),
    (56, 504, 56),
    (112, 1008, 112),
    (112, 1344, 112),
    (112, 3136, 112),
    (176, 4928, 176),
    (224, 2016, 224),
    (224, 12544, 224),
    (448, 50176, 448),
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
    python mapping_batchedfc.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_batchedfc.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_batchedfc.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
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
            I, O, groups = shape
            N = batch
            print("\n\nProblem size:")
            print(N, I, O, groups)
            layer_name = f"({N},{I},{O},{groups})"
            try:
                cost = run(
                    N,
                    I,
                    O,
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