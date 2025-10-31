import os
import numpy as np
import imageio
from stable_baselines3 import SAC
from panda_pickplace_env import PandaPickPlaceEnv


MODEL_PATH = "sac_panda_pickplace"    
VIDEO_DIR = "videos"                       
NUM_EVAL_EPISODES = 10              
MAX_STEPS_PER_EPISODE = 200             
os.makedirs(VIDEO_DIR, exist_ok=True)     

# 加载环境和模型
eval_env = PandaPickPlaceEnv(render_mode="rgb_array", max_steps=MAX_STEPS_PER_EPISODE, randomize=True)
model = SAC.load(MODEL_PATH)

success_count = 0

for ep in range(1, NUM_EVAL_EPISODES + 1):
    obs, _ = eval_env.reset()  
    frames = []   
    episode_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward

        frame = eval_env.render() 
        frames.append(frame)

        if terminated or truncated:
            break

    success = bool(info.get("success", False))
    if success:
        success_count += 1

        video_path = os.path.join(VIDEO_DIR, f"episode_{ep}_success.mp4")
        try:

            imageio.mimsave(video_path, frames, fps=30)
            print(f"第 {ep} 回合成功完成任务！视频已保存：{video_path}，总奖励 = {episode_reward:.2f}")
        except Exception as e:
            print(f"保存第 {ep} 回合视频时出错：", e)
    else:
        print(f"第 {ep} 回合未成功，总奖励 = {episode_reward:.2f}")

print(f"评估完成：成功 {success_count} / {NUM_EVAL_EPISODES} 回合")
eval_env.close()
