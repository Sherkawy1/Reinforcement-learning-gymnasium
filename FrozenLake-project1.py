import numpy as np
import gymnasium as gym

env = gym.make('FrozenLake-v1', is_slippery=True)

state_space = env.observation_space.n
action_space = env.action_space.n

q_table = np.zeros((state_space, action_space))

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.0005
min_epsilon = 0.01
episodes = 20000

GOAL_REWARD = 25
HOLE_PUNISHMENT = -23
STEP_PENALTY = -0.8

print("Starting training...")

for episode in range(episodes):
    state, _ = env.reset()
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        next_state, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if terminated and env_reward == 1.0:
            custom_reward = GOAL_REWARD
        elif terminated and env_reward == 0.0:
            custom_reward = HOLE_PUNISHMENT
        else:
            custom_reward = STEP_PENALTY

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])

        q_table[state, action] = old_value + alpha * (custom_reward + gamma * next_max - old_value)
        state = next_state

    epsilon = max(min_epsilon, epsilon - epsilon_decay)

env.close()
print("Training completed successfully!\n")

action_labels = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

num_test_episodes = 100
successes = 0
total_steps_all = 0
min_steps = None

for ep in range(num_test_episodes):
    test_env = gym.make('FrozenLake-v1', is_slippery=True)
    state, _ = test_env.reset()
    done = False
    steps = 0
    reward = 0

    while not done:
        action = np.argmax(q_table[state, :])
        next_state, reward, terminated, truncated, _ = test_env.step(action)
        steps += 1
        state = next_state
        done = terminated or truncated

    test_env.close()

    if reward == 1.0:
        successes += 1
        total_steps_all += steps
        if min_steps is None or steps < min_steps:
            min_steps = steps

print(f"Success rate over {num_test_episodes} episodes: {successes}/{num_test_episodes} ({successes / num_test_episodes * 100:.1f}%)")
if successes > 0:
    print(f"Average steps on successful episodes: {total_steps_all / successes:.2f}")
    print(f"Shortest successful path seen: {min_steps} steps")

print("\n--- Visual run of one episode ---")

test_env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human')
state, _ = test_env.reset()
done = False
total_steps = 0
path = [state]

while not done:
    action = np.argmax(q_table[state, :])
    next_state, reward, terminated, truncated, _ = test_env.step(action)

    total_steps += 1
    path.append(next_state)

    print(f"Step {total_steps}: direction {action_labels[action]} | from cell {state} to cell {next_state}")

    state = next_state
    done = terminated or truncated

test_env.close()

print("\n--- Result Report ---")
print(f"Path taken: {path}")
print(f"Total steps: {total_steps}")

if reward == 1.0:
    print("Reached the goal!")
else:
    print("Fell into a hole.")