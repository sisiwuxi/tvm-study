import numpy as np
from tvm import relay, runtime
# from tvm.relay import testing
import tvm
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor
from tvm.contrib.cuda_graph import cuda_graph_executor
import tvm.testing
from sisi import get_workload

def input_shape(mod):
    return [int(x) for x in mod["main"].checked_type.arg_types[0].shape]


def test_debug_graph_executor():
    if not tvm.testing.device_enabled("llvm"):
        print("Skip because llvm is not enabled")
        return
    # mod, params = relay.testing.synthetic.get_workload()
    # mod, params = relay.testing.sisi.get_workload()
    mod, params = get_workload()
    import pdb;pdb.set_trace()
    
    with relay.build_config(opt_level=3):
        complied_graph_lib = relay.build_module.build(mod, "llvm", params=params)
    data = np.random.uniform(-1, 1, size=input_shape(mod)).astype("float32")

    # raw api
    dev = tvm.cpu()
    try:
        gmod = complied_graph_lib["debug_create"]("default", dev)
    except:
        print("Skip because debug graph_executor not enabled")
        return
    set_input = gmod["set_input"]
    run = gmod["run"]
    get_output = gmod["get_output"]
    set_input("data", tvm.nd.array(data))
    run()
    out = get_output(0).numpy()
    # tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

    # debug graph executor wrapper
    debug_g_mod = debug_executor.GraphModuleDebug(
        complied_graph_lib["debug_create"]("default", dev),
        [dev],
        complied_graph_lib.get_graph_json(),
        None,
    )
    debug_g_mod.set_input("data", data)
    debug_g_mod.run()
    out = debug_g_mod.get_output(0).numpy()
    # tvm.testing.assert_allclose(out, verify(data), atol=1e-5)

if __name__ == "__main__":
    test_debug_graph_executor()