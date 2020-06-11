import gym
import gym_tetris


def random_agent(episodes=1000000):
    env = gym.make('Tetris-v0')
    env.reset()
    env.render()
    for e in range(episodes):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        env.render()
        print(done)
        if done:
            break


if __name__ == "__main__":
    random_agent()
