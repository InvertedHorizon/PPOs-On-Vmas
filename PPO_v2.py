# 导入 PyTorch 库
import torch

# 导入 tensordict 模块，用于处理张量字典
from tensordict.nn import TensorDictModule  # 一个张量字典模块的类
from tensordict.nn.distributions import NormalParamExtractor   
from torch import multiprocessing

# 导入数据收集模块
from torchrl.collectors import SyncDataCollector    # 一个数据收集器的类
from torchrl.data.replay_buffers import ReplayBuffer    # 一个回放缓冲区的类
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement  # 一个采样器的类
from torchrl.data.replay_buffers.storages import LazyTensorStorage  # 一个存储方式的类

# 导入环境相关模块
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# 导入多智能体网络模块
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# 导入损失函数相关模块
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# 导入工具库
torch.manual_seed(0)  # 设置随机种子，确保结果的可重复性
from matplotlib import pyplot as plt  # 用于绘制图形
from tqdm import tqdm  # 用于显示进度条

# 设备相关
is_fork = multiprocessing.get_start_method() == "fork"  # 检查多进程的启动方法是否为 fork
# 如果 GPU 可用且不是 fork 方法，则使用 GPU，否则使用 CPU
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
vmas_device = device  # 模拟器运行的设备，VMAS 可以在 GPU 上运行
print(device)


# 采样相关参数
frames_per_batch = 60000  # 每次训练迭代收集的团队帧数
n_iters = 400  # 采样和训练的迭代次数
total_frames = frames_per_batch * n_iters  # 总的帧数


# 训练相关参数
num_epochs = 40  # 每次训练迭代的优化步骤数，对于一次训练迭代收集的同一批数据，多次挖掘数据的特征
minibatch_size = 4096  # 每个优化步骤中使用的小批量数据的大小，一次epoch会对一批数据进行多次的小批量训练
lr = 3e-4  # 学习率
max_grad_norm = 1.0  # 梯度的最大范数


# PPO 算法相关参数
clip_epsilon = 0.2  # PPO 损失的裁剪值
gamma = 0.99  # 折扣因子
lmbda = 0.9  # 广义优势估计的 lambda 值
entropy_eps = 1e-4  # PPO 损失中熵项的系数


max_steps = 200  # 每个环境当中多智能体与环境交互的最多步数
num_vmas_envs = (
    frames_per_batch // max_steps
)  # 向量化环境的数量，frames_per_batch 应该可以被这个数整除
scenario_name = "transport"  # 场景名称
n_agents = 4  # 智能体的数量


# 创建 VmasEnv 环境
env = VmasEnv(
    scenario=scenario_name,  # 场景名称
    num_envs=num_vmas_envs,  # 环境数量
    continuous_actions=True,  # VMAS 支持连续和离散动作
    max_steps=max_steps,  # 最大步数
    device=vmas_device,  # 设备
    # 场景的自定义参数
    n_agents=n_agents,  # 每个 VMAS 场景不同的自定义参数，可查看 VMAS 仓库了解更多
)


print("action_spec:", env.full_action_spec)  # 打印动作规范
print("reward_spec:", env.full_reward_spec)  # 打印奖励规范
print("done_spec:", env.full_done_spec)  # 打印完成规范
print("observation_spec:", env.observation_spec)  # 打印观察规范


print("action_keys:", env.action_keys)  # 打印动作键
print("reward_keys:", env.reward_keys)  # 打印奖励键
print("done_keys:", env.done_keys)  # 打印完成键


# 对环境进行转换，将奖励求和并存储在新键中
env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)

# check_env_specs(env)  # 检查环境规范

# 主要用于渲染的时候采集数据用的
n_rollout_steps = 200  # 推出的步数，与环境进行交互的步数
rollout = env.rollout(n_rollout_steps)  # 进行环境推出操作，用于收集与环境进行交互一定步数后的数据
print("rollout of three steps:", rollout)  # 打印推出结果
print("Shape of the rollout TensorDict:", rollout.batch_size)  # 打印推出的 TensorDict 的形状

# share_parameters_policy = True

# 选择算法，这里可以是 "MAPPO", "CPPO", "IPPO"
algorithm = "IPPO"


# 根据选择的算法设置评论家网络和策略网络是否集中化
if algorithm == "MAPPO":
    centralised_critic = True    # 集中化的评论家网络
    centralised_policy = False   # 分散化的策略网络
elif algorithm == "IPPO":
    centralised_critic = False   # 分散化的评论家网络
    centralised_policy = False   # 分散化的策略网络
elif algorithm == "CPPO":
    centralised_critic = True    # 集中化的评论家网络
    centralised_policy = True    # 集中化的策略网络

# 定义策略网络
policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],  # 每个智能体的输入维度，从观察规范中获取
        n_agent_outputs=2 * env.action_spec.shape[-1],  # 每个智能体的输出维度，是动作维度的两倍
        n_agents=env.n_agents,  # 智能体的数量
        centralised=centralised_policy,  # 策略是否集中化
        share_params=True,  # 是否共享参数，智能体同质化时参数共享，观察空间、动作空间相同
        device=device,  # 设备
        depth=2,  # 网络的深度
        num_cells=256,  # 每层的单元数
        activation_class=torch.nn.Tanh,  # 激活函数
    ),
    NormalParamExtractor(),  # 将最后一个维度拆分为位置和非负尺度两个输出，提取成正态分布的参数
)


# 将策略网络包装在 TensorDictModule 中，这里通过指定键，来让模型去找到输入的数据来源，以及输出的数据去向
policy_module = TensorDictModule(
    policy_net,
    in_keys=[("agents", "observation")],  # 输入键
    out_keys=[("agents", "loc"), ("agents", "scale")],  # 输出键
)

# 创建概率性策略执行者
# 动作空间连续，
# 策略并不是直接通过网络来输出一个动作，而是输出一个正态分布的参数，然后再从正态分布当中采样一个动作
# 使用TanhNormal可以限制分布的范围
policy = ProbabilisticActor(
    module=policy_module,
    spec=env.unbatched_action_spec,  # 未批处理的动作规范
    in_keys=[("agents", "loc"), ("agents", "scale")],  # 输入键
    out_keys=[env.action_key],  # 输出键
    distribution_class=TanhNormal,  # 分布类，使用 TanhNormal 分布
    distribution_kwargs={
        "low": env.unbatched_action_spec[env.action_key].space.low,  # 分布的下限
        "high": env.unbatched_action_spec[env.action_key].space.high,  # 分布的上限
    },
    return_log_prob=True,  # 是否返回对数概率
    log_prob_key=("agents", "sample_log_prob"),  # 存储对数概率的键
)

# critic_net = MultiAgentMLP(
#     n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
#     n_agent_outputs=1,  # 每个智能体的输出为 1 个值
#     n_agents=env.n_agents,
#     centralised=mappo,
#     share_params=share_parameters_critic,
#     device=device,
#     depth=2,
#     num_cells=256,
#     activation_class=torch.nn.Tanh,
# )

# 定义评论家网络
critic_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    n_agent_outputs=1,
    n_agents=env.n_agents,
    centralised=centralised_critic,  # 评论家网络是否集中化
    share_params=True,  # 是否共享参数
    device=device,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)

# 将评论家网络包装在 TensorDictModule 中
critic = TensorDictModule(
    module=critic_net,
    in_keys=[("agents", "observation")],  # 输入键
    out_keys=[("agents", "state_value")],  # 输出键
)

print("Running policy:", policy(env.reset()))  # 打印运行策略的结果
print("Running value:", critic(env.reset()))  # 打印运行评论家的结果

# 创建数据收集器，用于为每一次的训练收集数据
collector = SyncDataCollector(
    env,
    policy,
    device=vmas_device,
    storing_device=device,
    frames_per_batch=frames_per_batch,  # 每批收集的帧数
    total_frames=total_frames,  # 总的帧数
)

# 创建重放缓冲区
# 从一批数据上采样小批量数据
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=device
    ),  # 存储每轮迭代收集的帧数，存储在指定设备上
    sampler=SamplerWithoutReplacement(),  # 不重复采样器
    batch_size=minibatch_size,  # 小批量的大小
)

# 创建 PPO 损失模块
# PPO损失包含三个部分，策略损失、价值损失、熵损失
# 策略损失旨在是策略朝着更优的方向更新，使智能体更倾向于选择具有更高优势的动作。策略通过策略网络得到
# 价值损失旨在使评论家网络的预测值与目标值之间的误差最小化。预测值通过价值网络得到，目标值通过广义优势估计得到
# 熵损失旨在使策略网络的输出更加均匀，避免过度拟合，用于鼓励策略的探索
# 裁剪值用于限制策略损失的变化范围，避免策略过于剧烈地更新
# 熵系数用于控制熵损失的权重，较高的熵系数会鼓励策略的探索，较低的熵系数会鼓励策略的利用
# 归一化优势可以避免在智能体维度上进行归一化，因为这会导致智能体之间的优势不平衡
loss_module = ClipPPOLoss(
    actor_network=policy,  # 策略网络
    critic_network=critic,  # 评论家网络
    clip_epsilon=clip_epsilon,  # 裁剪值
    entropy_coef=entropy_eps,  # 熵系数
    normalize_advantage=False,  # 避免在智能体维度上进行归一化
)

# 设置损失模块的键，告诉它在哪里找到相应的数据
loss_module.set_keys(
    reward=env.reward_key,
    action=env.action_key,
    sample_log_prob=("agents", "sample_log_prob"),
    value=("agents", "state_value"),
    # 以下两个键将扩展以匹配奖励的形状
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)


# 构建价值估计器，这里使用广义优势估计（GAE），用于估计目标价值
# 广义优势估计，当lambda=0时，GAE 退化为单步的优势估计，类似于 TD 误差；当lambda=1时，GAE 变成了蒙特卡洛回报的优势估计，具有较高的偏差但方差较小。通过调整 ，可以在偏差和方差之间找到一个平衡。
loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)
GAE = loss_module.value_estimator


# 创建优化器，使用 Adam 优化器优化损失模块的参数
optim = torch.optim.Adam(loss_module.parameters(), lr)

# 创建进度条
pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

episode_reward_mean_list = []  # 存储每回合奖励均值的列表
# 从数据收集器中获取数据
for tensordict_data in collector:
    # 扩展 done 和 terminated 的维度以匹配奖励的形状，这是价值估计器的要求
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done"))
      .unsqueeze(-1)
      .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )
    tensordict_data.set(
        ("next", "agents", "terminated"),
        tensordict_data.get(("next", "terminated"))
      .unsqueeze(-1)
      .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )

    # 计算 GAE 并添加到数据中，不计算梯度
    # ClipPPOLoss内部会自动根据critic_network复制一份得到target_critic_network，使用目标网络可以使训练更加稳定，尤其是对于包含时序差分的结构
    with torch.no_grad():
        GAE(
            tensordict_data,
            params=loss_module.critic_network_params,
            target_params=loss_module.target_critic_network_params,
        )

    # 将数据展平，以便打乱数据
    data_view = tensordict_data.reshape(-1)     # 展平张量字典当中的所有元素为一个一维张量

    # 将数据添加到重放缓冲区
    replay_buffer.extend(data_view)

    # 进行多次优化
    for _ in range(num_epochs):
        # 进行多次小批量优化
        for _ in range(frames_per_batch // minibatch_size):
            # 从重放缓冲区中采样小批量数据
            subdata = replay_buffer.sample()
            # 计算损失值
            loss_vals = loss_module(subdata)

            # 计算总损失，包括目标损失、评论家损失和熵损失
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # 反向传播
            loss_value.backward()

            # 裁剪梯度，防止梯度爆炸，这是可选的
            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )

            # 优化器更新参数
            optim.step()
            # 清空梯度
            optim.zero_grad()

    # 更新数据收集器中的策略权重
    collector.update_policy_weights_()

    # 获取所有达到完成状态的数据
    done = tensordict_data.get(("next", "agents", "done") )
    # 计算完成回合的平均奖励
    episode_reward_mean = (
        tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list.append(episode_reward_mean)
    # 更新进度条的描述
    pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
    pbar.update()

# 绘制训练结果
plt.plot(episode_reward_mean_list)
plt.xlabel("Training iterations")
plt.ylabel("Reward")
plt.title("Episode reward mean")
plt.show()

# 不计算梯度，使用策略在环境中进行推出操作，并渲染环境
with torch.no_grad():
    env.rollout(
        max_steps=max_steps,
        policy=policy,
        callback=lambda env, _: env.render(),  # 渲染环境的回调函数
        auto_cast_to_device=True,  # 自动转换数据到设备
        break_when_any_done=False,  # 当有完成时不中断推出操作
    )