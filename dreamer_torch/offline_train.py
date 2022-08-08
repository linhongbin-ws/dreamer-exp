from dreamer_torch.dreamer import Dreamer, make_env, count_steps, make_dataset
import argparse
import pathlib
import ruamel.yaml as yaml
from dreamer_torch import tools
import sys
import torch
torch.cuda.empty_cache() 
from tqdm import tqdm
import functools
import numpy as np
# from gym_suture.envs.wrapper import make_env as suture_make_env
to_np = lambda x: x.detach().cpu().numpy()

def main(config, traindir, savedir, train_reload_every, save_every, log_every, eval_every, epochs):
    config.savedir = pathlib.Path(savedir).expanduser()
    config.traindir = pathlib.Path(traindir).expanduser()
    config.steps //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    config.act = getattr(torch.nn, config.act)
    config.log_every = log_every
    


    print('traindir', config.traindir)
    print('savedir', config.savedir)
    config.savedir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    logger = tools.Logger(config.savedir, config.action_repeat * step)
    
    
    train_eps = None
    eval_eps = None
    make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps)
    train_envs = [make('train') for _ in range(config.envs)]
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]


    directory = config.traindir
    print(f"loading offline data {directory}")
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    train_dataset = make_dataset(train_eps, config)
    agent = Dreamer(config, logger, train_dataset).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (config.savedir / 'latest_model.pt').exists():
      print(f"load existing model: {config.savedir / 'latest_model.pt'}")
      agent.load_state_dict(torch.load(config.savedir / 'latest_model.pt'))
      agent._should_pretrain._once = False

    # if eval_every>0:
    #   config.evaldir = logdir / 'eval_eps'
    #   eval_eps = tools.load_episodes(config.evaldir, limit=1)
    #   env = suture_make_env('needle_picking')
    #   env.reset(reset_cam=True, is_random_init_pose=False)
    #   make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps, env)
    #   eval_envs = [make('eval') for _ in range(config.envs)]
    #   eval_dataset = make_dataset(eval_eps, config)

    pbar = tqdm(range(int(epochs)))
    for i in pbar:
      step = i 
      agent._step = step
      agent._train(next(agent._dataset))
      if agent._should_log(step):
        for name, values in agent._metrics.items():
          agent._logger.scalar(name, float(np.mean(values)))
          agent._metrics[name] = []
        openl = agent._wm.video_pred(next(agent._dataset))
        agent._logger.video('train_openl', to_np(openl))
        agent._logger.write(fps=True)
        
      if i % save_every == save_every-1: 
        torch.save(agent.state_dict(), config.savedir / 'latest_model.pt')

      # if (eval_every>0) and (i%eval_every==(eval_every-1)):
      #   eval_policy = functools.partial(agent, training=False)
      #   # print(f'[loop {i+1}] simulate 1 eps eval...')
      #   tools.simulate(eval_policy, eval_envs, episodes=5)
      #   # print(f'[loop {i+1}] evaluation.video prediction...')
      #   video_pred = agent._wm.video_pred(next(eval_dataset))
      #   logger.video('eval_openl', to_np(video_pred))
      
      if (train_reload_every>0) and (i%train_reload_every==(train_reload_every-1)):
        directory = config.traindir
        print(f"loading offline data {directory}")
        train_eps = tools.load_episodes(directory, limit=config.dataset_size)
        train_dataset = make_dataset(train_eps, config)
        agent._dataset = train_dataset
    # env.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', required=True)
  parser.add_argument('--traindir',  required=True)
  parser.add_argument('--savedir',  required=True)
  parser.add_argument("--train-reload-every", default=0)
  parser.add_argument('--save-every',  default=100) 
  parser.add_argument('--log-every',  default=1000)
  parser.add_argument("--eval-every", default=0)
  parser.add_argument("--epochs", default=1000000)
  
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
  main(config=parser.parse_args(remaining), 
      traindir=args.traindir,
      savedir=args.savedir,
      train_reload_every=int(args.train_reload_every),
      save_every=int(args.save_every), 
      log_every=int(args.log_every),
      eval_every=int(args.eval_every),
      epochs=int(args.epochs))

