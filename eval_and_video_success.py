import os
import numpy as np
import imageio
from stable_baselines3 import SAC
from panda_pickplace_env import PandaPickPlaceEnv  # 确保此模块在PYTHONPATH中

# 参数设置
MODEL_PATH = "sac_panda_pickplace"         # 训练好的模型路径（不带扩展名，会自动加载 .zip）
VIDEO_DIR = "videos"                       # 保存视频的文件夹
NUM_EVAL_EPISODES = 10                    # 评估回合数
MAX_STEPS_PER_EPISODE = 200               # 每回合最大步数，应与环境设置一致
os.makedirs(VIDEO_DIR, exist_ok=True)     # 创建保存视频的文件夹

# 加载环境和模型
eval_env = PandaPickPlaceEnv(render_mode="rgb_array", max_steps=MAX_STEPS_PER_EPISODE, randomize=True)
model = SAC.load(MODEL_PATH)

success_count = 0

for ep in range(1, NUM_EVAL_EPISODES + 1):
    obs, _ = eval_env.reset()   # 重置环境，随机摆放物体和目标
    frames = []                # 用于存储当前回合的图像帧
    episode_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # 使用训练好的模型预测动作
        # obs 是Gymnasium的新API返回的格式(obs, info)，这里只取obs；
        # 如果模型需要向量形式，可以使用obs[np.newaxis, :]扩展维度
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward

        # 收集当前帧图像
        frame = eval_env.render()  # 获取当前环境的RGB图像帧
        frames.append(frame)

        if terminated or truncated:
            break

    # 检查是否成功完成任务
    success = bool(info.get("success", False))
    if success:
        success_count += 1
        # 将帧列表保存为视频文件（MP4格式）
        video_path = os.path.join(VIDEO_DIR, f"episode_{ep}_success.mp4")
        try:
            # 使用imageio将帧列表保存为视频
            imageio.mimsave(video_path, frames, fps=30)
            print(f"第 {ep} 回合成功完成任务！视频已保存：{video_path}，总奖励 = {episode_reward:.2f}")
        except Exception as e:
            print(f"保存第 {ep} 回合视频时出错：", e)
    else:
        print(f"第 {ep} 回合未成功，总奖励 = {episode_reward:.2f}")

# 打印总体成功率
print(f"评估完成：成功 {success_count} / {NUM_EVAL_EPISODES} 回合")
eval_env.close()
