import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
from tvm import auto_scheduler
from tvm.relay.testing import layers
from tvm.relay.testing.resnet import residual_unit
from tvm.relay.testing.init import create_workload
# '''
def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler 更适合 NHWC 布局
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)

    return mod, params, input_shape, output_shape

# 定义神经网络和编译目标
network = "mobilenet"
batch_size = 1
# layout = "NHWC"
layout = "NCHW"
target = tvm.target.Target("cuda")
dtype = "float16"
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)

# 从网络中提取任务
print("Extract tasks...")
mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)

main_func = mod["main"]

# 想显示元数据则设置 show_meta_data=True
# print(mod.astext(show_meta_data=False))


tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)

def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=4800,  # 将此更改为 20000 以达到最佳性能
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2,
    )

    tuner.tune(tune_option)

# 不在网页服务器中运行调优，因为它需要的时间太长。
# 取消注释运行下面行。
run_tuning()

# 用历史最佳编译
print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

# 创建图执行器
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("data", data_tvm)

# 评估
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))
# '''
##############################################################
# Manually build the Resnet-18
# Auto-tensorize the conv2d operator
'''
def conv2d(N, C, H, W, K, R, S, stride, padding, dilation, dtype):
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype=dtype, name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype=dtype, name="B")

    Pad = tvm.te.compute(
        [N, C, pH, pW],
        lambda n, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(
                h >= padding, h - padding < H,
                w >= padding, w - padding < W),
            A[n, c, h - padding, w - padding],
            tvm.tir.const(0.0, A.dtype)
        ),
        name="Pad")

    rc = tvm.te.reduce_axis([0, C], name="rc")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    P = (pH - R) // stride + 1
    Q = (pW - S) // stride + 1
    Conv = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q:
            tvm.te.sum((Pad[n, rc, p+rr, q+rs] * B[k, rc, rr, rs]
                        ).astype(dtype), axis=[rc, rr, rs]),
        name="Conv"
    )
    return [A, B, Conv]



batch_size = 1
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
dtype="float32"
data_layout = "NHWC"
bn_axis = data_layout.index("C")
filter_list = [64, 64, 128, 256, 512]
kernel_layout = "OIHW" if data_layout == "NCHW" else "HWIO"

units = [2, 2, 2, 2]
num_unit = len(units)
num_stages = 4
assert num_unit == num_stages
bottle_neck = False
num_classes=1000
N, C, H, W, K, R, S, stride, padding, dilation = 1, 64, 56, 56, 512, 3, 3, 1, 1, 1

data = relay.var("data", shape=data_shape, dtype=dtype)
data = layers.batch_norm_infer(
    data=data, epsilon=2e-5, axis=bn_axis, scale=False, name="bn_data"
)
(_, _, height, _) = data_shape
if data_layout == "NHWC":
    (_, height, _, _) = data_shape

body = layers.conv2d(
    data=data,
    channels=filter_list[0],
    kernel_size=(3, 3),
    strides=(1, 1),
    padding=(1, 1),
    name="conv0",
    data_layout=data_layout,
    kernel_layout=kernel_layout,
)
# body = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
exit(1)
for i in range(num_stages):
    body = residual_unit(
        body,
        filter_list[i + 1],
        (1 if i == 0 else 2, 1 if i == 0 else 2),
        False,
        name=f"stage{i + 1}_unit1",
        bottle_neck=bottle_neck,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
    )
    for j in range(units[i] - 1):
        body = residual_unit(
            body,
            filter_list[i + 1],
            (1, 1),
            True,
            name=f"stage{i + 1}_unit{j + 2}",
            bottle_neck=bottle_neck,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
        )
bn1 = layers.batch_norm_infer(data=body, epsilon=2e-5, axis=bn_axis, name="bn1")
relu1 = relay.nn.relu(data=bn1)
# Although kernel is not used here when global_pool=True, we should put one
pool1 = relay.nn.global_avg_pool2d(data=relu1, layout=data_layout)
flat = relay.nn.batch_flatten(data=pool1)
fc1 = layers.dense_add_bias(data=flat, units=num_classes, name="fc1")
net = relay.nn.softmax(data=fc1)
net = relay.Function(relay.analysis.free_vars(net), net)
mod, params = create_workload(net)
# print(mod)
'''
##############################################################
