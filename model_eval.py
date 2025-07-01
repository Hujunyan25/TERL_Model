from policy.TERL_model import TERLConfig,TERLPolicy
import torch
config = TERLConfig(
    hidden_dim=256,
    num_heads=4,
    num_layers=4,
    action_size=9,
    num_quantiles=32,
    num_cosine_features=64,
    device='cpu',
    seed=0,
    learning_rate=1e-4,
    gradient_clip=1.0
)
model = TERLPolicy(config)

model_path = r"D:\TERL_eval_model\TERL\TrainedModels\TERL\network_params_v36.pth"
#model_path = "TrainedModels\TERL\network_params_v36.pth"
try:
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    print("模型加载成功")
except FileNotFoundError:
    print("未找到模型：{model_path}")
except Exception as e:
    print(f"加载模型出错：{e}")
    
