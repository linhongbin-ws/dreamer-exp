import argparse
import functools
import os
import pathlib
import sys
from dreamer_torch.dreamer import count_steps, Dreamer
from dreamer_torch import tools
from dreamer_torch import wrappers
from dreamer_torch import exploration
import ruamel.yaml as yaml
import torch
import numpy as np
from torch import distributions as torchd
from tqdm import tqdm
import time

def process_episode(config, directory, episode):
    filenames = tools.save_episodes(directory, [episode])
def count_eps(folder):
  return sum([1 for n in folder.glob('*.npz')])


def dreamer_agent_func(savedir, config, env):
  config.device = 'cpu'
  _agent = Dreamer(config, None, None, env, step=0).to('cpu')
  _agent.requires_grad_(requires_grad=False)
  agent_func = functools.partial(_agent, training=False)
  if (savedir / 'latest_model.pt').exists():
    while True:
      print(f"load existing {savedir / 'latest_model.pt'}")
      try:
        _agent.load_state_dict(torch.load(savedir / 'latest_model.pt'))
        _agent._should_pretrain._once = False
        break
      except Exception as e:
        print(e)
        print("try in 1 second")
        time.sleep(1)
  else:
    print(f"no pretrain model {savedir / 'latest_model.pt'}")
  return agent_func

def main(config, eps, savedir, seed, agent, reload_agent_param, mode='train'):
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    config.act = getattr(torch.nn, config.act)

    savedir = pathlib.Path(savedir).expanduser()
    print('savedir: ', savedir)
    traindir = savedir / {'train':'train_eps', 'test':'test_eps'}[mode] / agent  
    savedir.mkdir(parents=True, exist_ok=True)
    traindir.mkdir(parents=True, exist_ok=True)


    print('Create envs.')
    suite, task = config.task.split('_', 1)
    if config.task == 'ambf_np_discrete':
      from gym_suture.envs.wrapper import make_env as suture_make_env
      env = suture_make_env('ambf_needle_picking_64x64_discrete')
      env = wrappers.OneHotAction(env)
    elif config.task == 'suture':
      env = suture_make_env("needle_picking")
    elif suite == 'dmc':
      env = wrappers.DeepMindControl(task, config.action_repeat, config.size)
      env = wrappers.NormalizeActions(env)
    elif suite == 'atari':
      env = wrappers.Atari(
          task, config.action_repeat, config.size,
          grayscale=config.grayscale,
          life_done=False and ('train' in mode),
          sticky_actions=True,
          all_actions=True)
      env = wrappers.OneHotAction(env)
    elif suite == 'dmlab':
      env = wrappers.DeepMindLabyrinth(
          task,
          mode if 'train' in mode else 'test',
          config.action_repeat)
      env = wrappers.OneHotAction(env)
    else:
      raise NotImplementedError

    # 
    env.seed = seed
    env.reset()
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key='action')
    callbacks = [functools.partial(
            process_episode, config, traindir,)]
    env = wrappers.CollectDataset(env, callbacks)
    env = wrappers.RewardObs(env)
    
    
    train_envs = [env]
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]

    to_eps = max(0, eps - count_eps(traindir))
    print(f'collect eps: {eps}, existing {eps-to_eps}, to collect {to_eps} ')
    

    if agent == 'oracle':
      oracle_actor = exploration.OracleActor(env, noise_scale=config.oracle_noise_scale)
      # print(oracle_actor.loc, oracle_actor.scale)
      def agent_func(o, d, s, r):
          act = oracle_actor.actor()
          action = act.sample()
          if not oracle_actor.is_discrete:
            action = torch.clip(action, -torch.ones(action.shape), torch.ones(action.shape))
          logprob = act.log_prob(action)
          # print('action', action, 'logprob', logprob)
          return {'action': action, 'logprob': logprob}, None

    elif agent == 'random':
      if hasattr(acts, 'discrete'):
        random_actor = tools.OneHotDist(torch.zeros_like(torch.Tensor(acts.low))[None])
      else:
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(torch.Tensor(acts.low)[None],
                                  torch.Tensor(acts.high)[None]), 1)
      def agent_func(o, d, s, r):
        action = random_actor.sample()
        logprob = random_actor.log_prob(action)
        return {'action': action, 'logprob': logprob}, None
      
    elif agent == 'dreamer':
      agent_func = dreamer_agent_func(savedir, config, env)

    for i in tqdm(range(to_eps)):
      tools.simulate(agent_func, train_envs, episodes=1)
      if i % reload_agent_param==0 and agent=='dreamer':
        agent_func = dreamer_agent_func(savedir, config, env)
    
    if suite == 'ambf':
      env.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--eps', required=True)
  parser.add_argument('--configs', nargs='+', required=True)
  parser.add_argument('--savedir',required=True)
  parser.add_argument('--seed',default=1)
  parser.add_argument('--agent',default='random')
  parser.add_argument('--reload',default=2)
  parser.add_argument('--mode',default='train')
  args, remaining = parser.parse_known_args()
  configs = yaml.safe_load(
      (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  defaults = {}
  for name in args.configs:
    defaults.update(configs[name])
  parser = argparse.ArgumentParser()
  for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
  main(config=parser.parse_args(remaining), eps=int(args.eps), savedir=args.savedir, seed=int(args.seed), agent=args.agent, reload_agent_param=int(args.reload), mode=args.mode)
