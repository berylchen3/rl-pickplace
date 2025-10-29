# eval_and_video.py
import os
import imageio.v2 as imageio
from stable_baselines3 import SAC
from panda_pickplace_env import PandaPickPlaceEnv

def pick_model_path() -> str:
    """Prefer best model, then latest checkpoint, then final saved model."""
    # 1) Best model from EvalCallback
    best = "logs/best_model.zip"
    if os.path.exists(best):
        return best

    # 2) Latest checkpoint (largest <steps> number)
    ckpt_dir = "logs/ckpts"
    if os.path.isdir(ckpt_dir):
        zips = [f for f in os.listdir(ckpt_dir) if f.endswith(".zip")]
        if zips:
            # extract the integer '<steps>' from filenames like 'sac_panda_200000_steps.zip'
            def steps_num(name: str) -> int:
                parts = name.split("_")
                for i, p in enumerate(parts):
                    if p.isdigit() and i+1 < len(parts) and parts[i+1].startswith("steps"):
                        return int(p)
                # fallback: take the largest number appearing in the string
                nums = [int("".join(ch for ch in name if ch.isdigit()) or 0)]
                return max(nums) if nums else 0
            latest = max(zips, key=steps_num)
            return os.path.join(ckpt_dir, latest)

    # 3) Final model saved at the end of training
    final = "sac_panda_pickplace.zip"
    if os.path.exists(final):
        return final

    raise FileNotFoundError("No model file found in logs/ or project root.")

# ---- Load model (auto selection) ----
model_path = pick_model_path()
print("Loading model from:", model_path)
model = SAC.load(model_path)

# ---- Render one episode and export MP4 ----
env = PandaPickPlaceEnv(render_mode="rgb_array", max_steps=200, randomize=True)
obs, info = env.reset()

frames = []
success = False
for t in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()  # rgb_array frames (since render_mode="rgb_array")
    if frame is not None:
        frames.append(frame)
    if terminated or truncated:
        success = success or info.get("success", False)
        break



env.close()

# Save as MP4 (you can change to .gif if you prefer)
out_path = "panda_pickplace.mp4"
writer = imageio.get_writer(out_path, fps=30)
for f in frames:
    writer.append_data(f)
writer.close()
print(f"Saved to {out_path} | success = {success}")


# # eval_and_video.py - Robust evaluation with error handling
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"

# import numpy as np
# import imageio.v2 as imageio
# from stable_baselines3 import SAC
# from panda_pickplace_env import PandaPickPlaceEnv
# import time

# def pick_model_path() -> str:
#     """Prefer best model, then latest checkpoint, then final saved model."""
#     # 1) Best model from EvalCallback
#     best = "logs/best_model.zip"
#     if os.path.exists(best):
#         return best

#     # 2) Latest checkpoint (largest <steps> number)
#     ckpt_dir = "logs/ckpts"
#     if os.path.isdir(ckpt_dir):
#         zips = [f for f in os.listdir(ckpt_dir) if f.endswith(".zip")]
#         if zips:
#             def steps_num(name: str) -> int:
#                 parts = name.split("_")
#                 for i, p in enumerate(parts):
#                     if p.isdigit() and i+1 < len(parts) and parts[i+1].startswith("steps"):
#                         return int(p)
#                 nums = [int("".join(ch for ch in name if ch.isdigit()) or 0)]
#                 return max(nums) if nums else 0
#             latest = max(zips, key=steps_num)
#             return os.path.join(ckpt_dir, latest)

#     # 3) Final model saved at the end of training
#     final = "sac_panda_pickplace_final.zip"
#     if os.path.exists(final):
#         return final
    
#     final2 = "sac_panda_pickplace.zip"
#     if os.path.exists(final2):
#         return final2

#     raise FileNotFoundError("No model file found in logs/ or project root.")

# def evaluate_episodes(model, num_episodes=3, max_steps=250, randomize=True, record_video=True):
#     """
#     Evaluate model for multiple episodes and optionally record video.
    
#     Returns:
#         frames: List of RGB frames (if record_video=True)
#         stats: Dictionary with evaluation statistics
#     """
#     all_frames = []
#     all_rewards = []
#     all_successes = []
#     all_lengths = []
    
#     for ep in range(num_episodes):
#         print(f"\n{'='*60}")
#         print(f"Episode {ep+1}/{num_episodes}")
#         print(f"{'='*60}")
        
#         # Create fresh environment for each episode
#         env = PandaPickPlaceEnv(
#             render_mode="rgb_array" if record_video else None,
#             max_steps=max_steps,
#             randomize=randomize
#         )
        
#         obs, info = env.reset()
#         episode_frames = []
#         episode_reward = 0.0
#         success = False
        
#         for t in range(max_steps):
#             # Get action from model
#             action, _ = model.predict(obs, deterministic=True)
            
#             # Step environment
#             obs, reward, terminated, truncated, info = env.step(action)
#             episode_reward += reward
            
#             # Render frame with error handling
#             if record_video:
#                 try:
#                     frame = env.render()
#                     if frame is not None and isinstance(frame, np.ndarray):
#                         # Validate frame dimensions
#                         if frame.shape == (480, 640, 3):
#                             episode_frames.append(frame)
#                         else:
#                             print(f"  Warning: Invalid frame shape {frame.shape}, skipping")
#                 except Exception as e:
#                     print(f"  Warning: Render error at step {t}: {e}")
#                     # Continue without this frame
            
#             # Print progress every 50 steps
#             if (t + 1) % 50 == 0:
#                 print(f"  Step {t+1}/{max_steps} | Reward: {episode_reward:.1f} | "
#                       f"Lifted: {info.get('lifted', False)} | "
#                       f"Dist to goal: {info.get('dist_obj_goal', 0):.3f}")
            
#             # Check termination
#             if terminated or truncated:
#                 success = info.get("success", False)
#                 break
        
#         # Episode summary
#         print(f"\n  Results:")
#         print(f"    Duration: {t+1} steps")
#         print(f"    Total reward: {episode_reward:.2f}")
#         print(f"    Success: {'✓ YES' if success else '✗ NO'}")
#         print(f"    Frames captured: {len(episode_frames)}")
        
#         # Store results
#         all_frames.extend(episode_frames)
#         all_rewards.append(episode_reward)
#         all_successes.append(success)
#         all_lengths.append(t + 1)
        
#         # Cleanup
#         env.close()
#         del env
#         time.sleep(0.5)  # Brief pause between episodes
    
#     # Overall statistics
#     stats = {
#         "num_episodes": num_episodes,
#         "mean_reward": np.mean(all_rewards),
#         "std_reward": np.std(all_rewards),
#         "success_rate": np.mean(all_successes),
#         "mean_length": np.mean(all_lengths),
#         "total_frames": len(all_frames),
#     }
    
#     return all_frames, stats

# def main():
#     print("="*60)
#     print("Panda Pick-and-Place Evaluation")
#     print("="*60)
    
#     # Load model
#     try:
#         model_path = pick_model_path()
#         print(f"\n📦 Loading model from: {model_path}")
#         model = SAC.load(model_path)
#         print("✓ Model loaded successfully")
#     except Exception as e:
#         print(f"\n❌ Error loading model: {e}")
#         return
    
#     # Configuration
#     NUM_EPISODES = 3
#     MAX_STEPS = 250
#     RANDOMIZE = False  # Set to True for varied scenarios
#     OUTPUT_VIDEO = "panda_pickplace_demo.mp4"
#     FPS = 30
    
#     print(f"\n⚙️  Configuration:")
#     print(f"   Episodes: {NUM_EPISODES}")
#     print(f"   Max steps per episode: {MAX_STEPS}")
#     print(f"   Randomize positions: {RANDOMIZE}")
#     print(f"   Output video: {OUTPUT_VIDEO}")
    
#     # Run evaluation
#     print(f"\n🎬 Starting evaluation...")
#     try:
#         frames, stats = evaluate_episodes(
#             model=model,
#             num_episodes=NUM_EPISODES,
#             max_steps=MAX_STEPS,
#             randomize=RANDOMIZE,
#             record_video=True
#         )
#     except Exception as e:
#         print(f"\n❌ Evaluation error: {e}")
#         import traceback
#         traceback.print_exc()
#         return
    
#     # Save video
#     if len(frames) > 0:
#         print(f"\n💾 Saving video...")
#         try:
#             writer = imageio.get_writer(OUTPUT_VIDEO, fps=FPS)
#             for frame in frames:
#                 writer.append_data(frame)
#             writer.close()
            
#             duration = len(frames) / FPS
#             print(f"✓ Video saved: {OUTPUT_VIDEO}")
#             print(f"  Frames: {len(frames)}")
#             print(f"  Duration: {duration:.1f} seconds")
#         except Exception as e:
#             print(f"❌ Error saving video: {e}")
#     else:
#         print(f"\n⚠️  No frames captured - cannot create video")
    
#     # Print summary
#     print(f"\n{'='*60}")
#     print("Evaluation Summary")
#     print(f"{'='*60}")
#     print(f"Episodes evaluated: {stats['num_episodes']}")
#     print(f"Mean reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
#     print(f"Success rate: {stats['success_rate']*100:.1f}%")
#     print(f"Mean episode length: {stats['mean_length']:.1f} steps")
#     print(f"{'='*60}")

# if __name__ == "__main__":
#     main()
