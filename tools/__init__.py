# 旧版 test / 预训练 / 微调 runner 依赖 cv2/sklearn/matplotlib 等重型可选依赖。
# 语义分割 S0/S1/S2 只用 tools.runner_seg，不需要这些；在未安装它们的机器（如 c135）
# 上容忍跳过，避免 tools 包整体 import 失败。安装了这些依赖的机器行为不变。
try:
    from .runner import test_net
except ImportError:
    test_net = None

try:
    from .runner_pretrain import run_net as pretrain_run_net
except ImportError:
    pretrain_run_net = None

try:
    from .runner_finetune import run_net as finetune_run_net
    from .runner_finetune import test_net as test_run_net
except ImportError:
    finetune_run_net = None
    test_run_net = None
