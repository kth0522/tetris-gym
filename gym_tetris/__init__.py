from gym.envs.registration import register

register(id='Tetris-v0',
         entry_point='gym_tetris.envs:TetrisEnv',
)