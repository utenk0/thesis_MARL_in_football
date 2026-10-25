# Initialise the football environment.
football_env = FootballEnvWrapper(num_per_team=11)
# Initialise the teams.
rl_team = RLTeam()
random_team = RandomTeam()
teams = [rl_team, random_team]

while True:
    # Reset the agents' internal states.
    for t_i in range(len(teams)):
        teams[t_i].reset_brain()

    # Start a new game.
    observations, states, rewards = football_env.reset_game()

    # Store the initial observations, states and rewards received
    # from the environment.
    rl_team.observe_first(observations[0], states[0], rewards[0])

    done = False
    while not done:
        # Get the agents' actions.
        actions = []
        for t_i in range(len(teams)):
            actions.append(teams[t_i].get_team_actions(observations[t_i]))

        # Step the environment.
        observations, states, rewards, done = football_env.step(actions)

        # Display the football screen.
        football_env.display_screen()

        # Store the observations, states and rewards for
        # environment step.
        rl_team.observe(observations[0], states[0], actions[0], rewards[0], done)

        # Perform an agent update step.
        rl_team.trainer_step()
