from gymnasium.envs.registration import register

register(
    id='Othello-v0',
    entry_point='gym_games.envs.OthelloEnv:OthelloEnv',
    max_episode_steps=300,
)