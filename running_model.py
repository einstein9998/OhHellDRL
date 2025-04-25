import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import torch

from ohhell import OhHellEnv
from masking import MaskedMlpPolicy

env = OhHellEnv()
env = make_vec_env(OhHellEnv)

eval_env = OhHellEnv()
eval_env = make_vec_env(OhHellEnv)

checkpoint_callback = CheckpointCallback(save_freq=6144, save_path='./logs/logs_ppo/legal_moves', name_prefix='training')

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/logs_ppo/legal_moves/best_model', log_path='./logs/logs_ppo/legal_moves/', eval_freq=36864,
                             deterministic=True, render=False)

callback = CallbackList([checkpoint_callback, eval_callback])

policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=(dict(pi=[350, 350, 63], vf=[350, 350, 100])))

model = PPO(MaskedMlpPolicy, env, policy_kwargs=policy_kwargs, tensorboard_log="./tmp/", 
             verbose=1, seed=2)

start = time.time()
model.learn(total_timesteps=12000000, callback=callback)
end = time.time()
print("Time Taken: %f" % (end-start))        

# Saving the model to the current working directory
model.save("ppo_ohhell_legal_moves")