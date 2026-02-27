import gym
import gym_super_mario_bros

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    while hasattr(env, 'env'):
        env = env.env
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        total_reward += reward
        print(f"Reward: {reward}, Info: {info}")
    env.close()
    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    main()
