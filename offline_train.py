from dream_suture.dreamer import Dreamer, make_env, count_steps, make_dataset
import argparse
import pathlib
import ruamel.yaml as yaml
from dream_suture import tools
import sys
import torch
torch.cuda.empty_cache() 
from tqdm import tqdm
import functools
from gym_suture.envs.wrapper import make_env as suture_make_env
to_np = lambda x: x.detach().cpu().numpy()

def main(config, save_every, traindir, log_every, eval_every, traindir_every):
    logdir = pathlib.Path(traindir).expanduser()
    config.traindir = logdir / 'train_eps'
    config.steps //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat
    config.act = getattr(torch.nn, config.act)

    config.train_every = 1
    config.log_every = log_every

    print('traindir', config.traindir)
    print('logdir', logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)


    directory = config.traindir
    print(f"loading offline data {directory}")
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    train_dataset = make_dataset(train_eps, config)
    agent = Dreamer(config, logger, train_dataset, env=None).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / 'latest_model.pt').exists():
      print(f"load existing model: {logdir / 'latest_model.pt'}")
      agent.load_state_dict(torch.load(logdir / 'latest_model.pt'))
      agent._should_pretrain._once = False

    if eval_every>0:
      config.evaldir = logdir / 'eval_eps'
      eval_eps = tools.load_episodes(config.evaldir, limit=1)
      env = suture_make_env('needle_picking')
      env.reset(reset_cam=True, is_random_init_pose=False)
      make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps, env)
      eval_envs = [make('eval') for _ in range(config.envs)]
      eval_dataset = make_dataset(eval_eps, config)

    pbar = tqdm(range(int(config.offline_training_steps)))
    for i in pbar:
      agent(obs=None, reset=[True], state=None, reward=None, training=True, inference=False)
      if i % save_every == save_every-1: 
        torch.save(agent.state_dict(), logdir / 'latest_model.pt')

      if (eval_every>0) and (i%eval_every==(eval_every-1)):
        eval_policy = functools.partial(agent, training=False)
        # print(f'[loop {i+1}] simulate 1 eps eval...')
        tools.simulate(eval_policy, eval_envs, episodes=5)
        # print(f'[loop {i+1}] evaluation.video prediction...')
        video_pred = agent._wm.video_pred(next(eval_dataset))
        logger.video('eval_openl', to_np(video_pred))
      
      if (traindir_every>0) and (i%traindir_every==(traindir_every-1)):
        directory = config.traindir
        print(f"loading offline data {directory}")
        train_eps = tools.load_episodes(directory, limit=config.dataset_size)
        train_dataset = make_dataset(train_eps, config)
        agent._dataset = train_dataset
    env.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', required=True)
  parser.add_argument('--save-every',  default=10) 
  parser.add_argument('--log-every',  default=10)
  parser.add_argument('--traindir',  required=True)
  parser.add_argument("--eval-every", default=0)
  parser.add_argument("--traindir-every", default=100)
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
  main(parser.parse_args(remaining), 
                        save_every=int(args.save_every), 
                        traindir=args.traindir, 
                        log_every=int(args.log_every),
                        eval_every=int(args.eval_every),
                        traindir_every=int(args.traindir_every))

