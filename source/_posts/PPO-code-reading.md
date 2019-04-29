---
title: PPO算法原理及代码阅读
date: 2018-10-06 15:50:20
tags: 
	- 深度强化学习
	- PPO
description: 介绍PPO算法原理，并阅读PPO的实现代码
mathjax: true
---
## 简介

继策略梯度方法后，PPO成为openAI力推的深度强化学习算法，在收敛性和稳定性上更优于之前的DRL方法。本文首先简单介绍PPO算法的原理，然后对 https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/pytorch-a2c-ppo-acktr 开源代码中实现PPO的部分开展详细的介绍。



## PPO算法原理

### Motivation
Policy gradient方法的步长（learning rate）不好确定，太大导致更新太快、不稳定且不易收敛，太小导致收敛速度过慢。因此通过计算new policy和old policy 的占比增加约束条件，限制policy的更新幅度。



### 算法结构和流程
actor-critic结构，两个loss--$J_{ppo}$和$L$，actor的目标是最大化前者，critic的目标是最小化后者。$J_{ppo}$的意义是，advantage(TD error)表示新策略value与旧策略value的差别，因此当advantage更大时，表示更新policy的幅度更大，让new policy发生的可能性更大；加上KL散度的约束，即限制了new policy与old policy之间的差距，保证收敛性；因此需要最大化$J_{ppo}$。$L$的意义即是TD error，最小化error使得value function的近似更加准确。
对于actor，优化的目标函数有几种形式：

$$L^{CLIP}(\theta)=\hat{E}_t[min(r_t(\theta)\hat{A}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

![目标函数$L^{CLIP}$](PPO-code-reading/L_CLIP.PNG)



考虑到更新的new policy与old policy之间额差别限制，在目标函数中增加KL散度作为penalty项：
$$L^{KLPEN}(\theta)=\hat{E}_t[r_t(\theta)\hat{A}_t-\beta  KL[\pi_{\theta_{old}}, \pi_{\theta_{new}}]]$$

### 单线程PPO

用$\pi_{\theta_{old}}$与环境交互T步，得到t=1,2,...,T对应的advantage。根据surrogate loss function $J_{ppo}$，用梯度法更新actor(policy)的参数$\theta$；根据TD-error用梯度法更新critic(value function)的参数$\phi$。

### 多线程PPO

多线程PPO相比于单线程PPO来说，区别在于rollouts中样本的来源是单个worker与单个环境交互，还是多个workers分别与多个环境同时交互。总的流程要点如下：
- 有一个共同的actor-critic和ppo agent以及storage，有多个workers；
- 这些workers平行地在不同环境中收集数据，并根据workers分类存到storage中；
- ppo agent根据从storage中采样得到的样本对actor-critic参数进行更新
- 参数更新之后，workers用新的actor-critic继续采集数据，重复以上更新-采集-更新的流程，得到最终的actor-critic模型。

算法伪代码如下图所示（截图自Google DeepMind的论文[Emergency of locomotion behaviours in rich environments](https://arxiv.org/pdf/1707.02286.pdf)）

![多进程PPO算法伪代码](PPO-code-reading/PPO_algo.png)




## 源码框架

这份代码实现了多线程的PPO，接下来将从整体框架、逻辑结构以及核心部分具体功能及实现来展开介绍。

### 目录框架
- arbitrary files
    - arguments.py
    - distributions.py
    - logger.py：调用tf.Summary函数，用容器存储训练过程数据
    - visualize.py：调用visdom可视化训练过程曲线
    - envs.py：调用baselines.bench.Monitor包装实验环境
- major files
    - ==main.py==：训练
    - ==storage.py==：存储rollouts数据
    - ==algo/PPO.py==：PPO算法，参数更新
    - model.py：搭建actor和critic的神经网络
    - enjoy.py：测试

### main.py理解逻辑结构

- 初始化

	多线程创建实验环境
	```python
	envs = [make_env(args.env_name, args.seed, i, args.log_dir, args.add_timestep) for i in range(args.num_processes)]
	```
	创建actor_critic，搭建agent的value function和policy模型
	```python
	actor_critic = Policy(obs_shape, envs.action_space, args.recurrent_policy, hasattr(tmp_env.unwrapped, 'N') and tmp_env.unwrapped.N or 1)
	```
	引入PPO算法，更新模型以及rollouts数据
	```python
	agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
					args.value_loss_coef, args.entropy_coef, lr=args.lr,
					eps=args.eps, max_grad_norm=args.max_grad_norm)
	```
	创建rollouts，存储元素包括observations, states, rewards, actions, value\_preds, returns, action\_log\_probs, masks，其中value\_preds是...，action\_log\_probs是新策略和久策略的可能性占比，returns是value function的值，每一个元素的数据结构为(num\_steps, num\_processes,1)的矩阵。
	```python
	rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
	```

- 与环境的交互，收集数据

  首先，rollouts中初始的observations为初始环境中agent获得的observation。对于一个agent（即一个线程），它的模型根据rollouts中的数据做出决策，
  ```python
  value, action, action_log_prob, states = actor_critic.act(
  						rollouts.observations[step],
  						rollouts.states[step],
  						rollouts.masks[step])
  ```
  得到的action用于环境交互更新环境状态及agent的observation，并将新的数据更新到rollouts中。

  ```python
  with torch.no_grad():
  	next_value = actor_critic.get_value(rollouts.observations[-1],
  										rollouts.states[-1],
  										rollouts.masks[-1]).detach()
  
  rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
  ```

- PPO优化参数
	agent根据更新后的rollouts中的数据，更新模型参数。
	```python
	value_loss, action_loss, dist_entropy = agent.update(rollouts)

	rollouts.after_update()
	```

### 核心部分的函数组成
上面介绍完PPO算法实现的具体流程以及核心部分担当的“角色”，接下来就进一步具体说明核心部分包括哪些成员函数去实现各自的功能。
- PPO.py
```python
class PPO(object):
    def __init__(self, actor_critic, <relative hyper-parameters ...>):
        # 初始化超参数，调用actor-critic
        ...
    def update(self, rollouts):
        # 从rollouts获取样本，根据actor-critic计算的value loss和action loss更新actor-critic参数
        ...
```

- model.py
```python
class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, recurrent_policy, num_agents=1):
        #选择value function和policy的网络结构
        ...
    def act(self, inputs, states, masks, deterministic=False):
        # 输出action，states，对应的value值，以及action占比
        ...
    def get_value(self, inputs, states, masks):
        # 获取value值
        ...
    def evaluate_actions(self, inputs, states, masks, action):
        # 评估action，输出value和action占比
        ...

# 不同网络结构的定义
class MLPBase(nn.Module):
    ...
class CNNBase(nn.Module):
    ...
class RNNBase(nn.Module):
    ...
```

actor的实际输出是以神经网络输出作为均值、以$delta$作为方差生成正态分布，在deterministic为False的时候，采样得到action；在deterministic为True的时候，直接以均值作为action。



- storage.py
```python
class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size):
        # 初始化rollouts记录的元素，torch.zeros(x,y,z)三维矩阵的数据结构
        ...
    def cuda(self):
        # cpu转gpu
        ...
    def insert(self, current_obs, state, action, action_log_prob, value_pred, reward, mask):
        # 更新rollouts存储的元素数据
        ...
    def after_update(self):
        # 将最新的元素数据放在初始位置
        ...
    def compute_returns(self, next_value, use_gae, gamma, tau):
        # 计算return
        ...
    def feed_forward_generator(self, advantages, num_mini_batch):
        # 迭代返回抽样样本
        ...
    def recurrent_generator(self, advantages, num_mini_batch):
        # 迭代返回抽样样本
        ...
```

feed\_forward\_generator的采样过程比较简单，下面具体看看用于RNN的样本生成器recurrent_generator的实现。

```python
def recurrent_generator(self, advantages, num_mini_batch):
		num_processes = self.rewards.size(1)
		assert num_processes >= num_mini_batch, (
			f"PPO requires the number processes ({num_processes}) "
			f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch}).")
		num_envs_per_batch = num_processes // num_mini_batch
		perm = torch.randperm(num_processes)
		for start_ind in range(0, num_processes, num_envs_per_batch):
			observations_batch = []
			states_batch = []
			actions_batch = []
			return_batch = []
			masks_batch = []
			old_action_log_probs_batch = []
			adv_targ = []
		for offset in range(num_envs_per_batch):
			ind = perm[start_ind + offset]
			observations_batch.append(self.observations[:-1, ind])
			states_batch.append(self.states[0:1, ind])
			actions_batch.append(self.actions[:, ind])
			return_batch.append(self.returns[:-1, ind])
			masks_batch.append(self.masks[:-1, ind])
			old_action_log_probs_batch.append(self.action_log_probs[:, ind])
			adv_targ.append(advantages[:, ind])

		#observations_batch = torch.cat(observations_batch, 0)
		#states_batch = torch.cat(states_batch, 0)
		#actions_batch = torch.cat(actions_batch, 0)
		#return_batch = torch.cat(return_batch, 0)
		#masks_batch = torch.cat(masks_batch, 0)
		#old_action_log_probs_batch = torch.cat(old_action_log_probs_batch, 0)
		#adv_targ = torch.cat(adv_targ, 0)

		T, N = self.num_steps, num_envs_per_batch
		# These are all tensors of size (T, N, -1)
		observations_batch = torch.stack(observations_batch, 1)
		actions_batch = torch.stack(actions_batch, 1)
		return_batch = torch.stack(return_batch, 1)
		masks_batch = torch.stack(masks_batch, 1)
		old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
		adv_targ = torch.stack(adv_targ, 1)

		# States is just a (N, -1) tensor
		states_batch = torch.stack(states_batch, 1).view(N, -1)

		# Flatten the (T, N, ...) tensors to (T * N, ...)
		observations_batch = _flatten_helper(T, N, observations_batch)
		actions_batch = _flatten_helper(T, N, actions_batch)
		return_batch = _flatten_helper(T, N, return_batch)
		masks_batch = _flatten_helper(T, N, masks_batch)
		old_action_log_probs_batch = _flatten_helper(T, N, old_action_log_probs_batch)
		adv_targ = _flatten_helper(T, N, adv_targ)

		yield observations_batch, states_batch, actions_batch, \
			return_batch, masks_batch, old_action_log_probs_batch, adv_targ
```

