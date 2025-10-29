# train_sac.py  (Improved: domain randomization + better hyperparams)
import os
# CRITICAL: Set threading limits BEFORE importing any libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import torch
# Force PyTorch to single thread (prevents conflicts with PyBullet)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from panda_pickplace_env import PandaPickPlaceEnv
import gc

# Custom callback to prevent memory leaks
class MemoryCleanupCallback(BaseCallback):
    """Periodically trigger garbage collection to prevent memory leaks"""
    def __init__(self, cleanup_freq=10000, verbose=0):
        super().__init__(verbose)
        self.cleanup_freq = cleanup_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.cleanup_freq == 0:
            gc.collect()
            if self.verbose > 0:
                print(f"[GC] Memory cleanup at step {self.n_calls}")
        return True

# ============ CONFIGURATION ============
NUM_ENVS   = 1           # Safe for Windows with DummyVecEnv
MAX_STEPS  = 250         # Increased from 220 to allow time for placement
TOTAL_STEPS = 400_000  # Increased from 1M - full pick-place needs more training

# Training with randomization, evaluation without
TRAIN_RANDOMIZE = True   # CRITICAL: Enable domain randomization for training
EVAL_RANDOMIZE  = False  # Keep eval deterministic for fair comparison

def make_env(seed_offset=0, randomize=True):
    """Factory function to create environment instances"""
    def _init():
        env = PandaPickPlaceEnv(
            render_mode=None, 
            max_steps=MAX_STEPS, 
            randomize=randomize  # Pass randomization flag
        )
        env.reset(seed=42 + seed_offset)
        return env
    return _init

if __name__ == "__main__":
    set_random_seed(42)
    
    print("=" * 60)
    print("Training Configuration:")
    print(f"  - Total timesteps: {TOTAL_STEPS:,}")
    print(f"  - Parallel envs: {NUM_ENVS}")
    print(f"  - Episode length: {MAX_STEPS}")
    print(f"  - Domain randomization: {TRAIN_RANDOMIZE}")
    print("=" * 60)

    # Training env WITH randomization (learns generalization)
    train_env = DummyVecEnv([make_env(i, randomize=TRAIN_RANDOMIZE) for i in range(NUM_ENVS)])
    
    # Evaluation env WITHOUT randomization (consistent metrics)
    eval_env = PandaPickPlaceEnv(
        render_mode=None, 
        max_steps=MAX_STEPS, 
        randomize=EVAL_RANDOMIZE
    )
    eval_env.reset(seed=999)  # Fixed seed for eval consistency

    # Evaluate every 20k environment steps
    eval_every_env_steps = 20_000
    eval_freq = max(1, eval_every_env_steps // NUM_ENVS)

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=eval_freq,
        save_path="./logs/ckpts/",
        name_prefix="sac_panda",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Memory cleanup callback to prevent leaks
    memory_cb = MemoryCleanupCallback(cleanup_freq=10000, verbose=1)

    # SAC hyperparameters (tuned for manipulation tasks)
    policy_kwargs = dict(
        net_arch=[256, 256]  # Increased from [128,128] for better capacity
    )
    
    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,          # Increased from 128
        gamma=0.99,              # Increased from 0.98 (care about future more)
        tau=0.005,               # Decreased from 0.02 (slower target update)
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/tb/",
        learning_starts=10_000,  # Start learning after 10k steps (build diverse buffer)
    )

    print("\nStarting training...")
    print("Monitor progress: tensorboard --logdir ./logs/tb/\n")

    try:
        model.learn(
            total_timesteps=TOTAL_STEPS, 
            callback=[eval_cb, ckpt_cb, memory_cb],  # Added memory cleanup
            progress_bar=False  # Set to True if tqdm/rich are installed
        )
        model.save("sac_panda_pickplace_final")
        print("\n✓ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        model.save("sac_panda_pickplace_interrupted")
    
    finally:
        train_env.close()
        eval_env.close()

    # Print training summary
    print("\n" + "=" * 60)
    print("Training Summary:")
    print("=" * 60)
    
    try:
        data = np.load("logs/evaluations.npz")
        ts = data["timesteps"]
        mean_r = data["results"].mean(axis=1)
        std_r = data["results"].std(axis=1)
        
        print(f"\nEvaluations performed: {len(ts)}")
        print(f"Best mean reward: {mean_r.max():.2f}")
        print(f"Final mean reward: {mean_r[-1]:.2f} ± {std_r[-1]:.2f}")
        
        # Check success rate (reward > 50 usually indicates success)
        success_evals = (mean_r > 50).sum()
        print(f"Successful evaluations: {success_evals}/{len(mean_r)}")
        
        print("\nDetailed evaluation history:")
        for i, (t, r, s) in enumerate(zip(ts, mean_r, std_r)):
            print(f"  Step {t:>7,}: {r:>6.2f} ± {s:>5.2f}")
            
    except FileNotFoundError:
        print("⚠ Evaluation data not found (logs/evaluations.npz)")
    
    print("=" * 60)