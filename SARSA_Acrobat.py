import numpy as np
import gym
import matplotlib.pyplot as plt

theta_space =     np.linspace(-1,-1,10)
theta_dot_space = np.linspace(-5,5,10)

def get_state(observation):
    cos_theta1, sin_theta1, cos_theta2, sin_theta2, theta1_dot, theta2_dot = observation
    c_th1 = int(np.digitize(cos_theta1, theta_space))
    s_th1 = int(np.digitize(sin_theta1, theta_space))
    c_th2 = int(np.digitize(cos_theta2, theta_space))
    s_th2 = int(np.digitize(sin_theta2, theta_space))
    th1_dot = int(np.digitize(theta1_dot, theta_dot_space))
    th2_dot = int(np.digitize(theta2_dot, theta_dot_space))
    return (c_th1, c_th2, s_th1, s_th2, th1_dot, th2_dot)

def max_action(Q, state, actions = [0,1,2]):
    values = np.array([Q[state, a] for a in actions])
    action = np.argmax(values)
    return action

if __name__ == '__main__':
    env = gym.make('Acrobot-v1')
    n_games = 50000
    alpha = 0.1
    gamma = 0.99
    eps = 1

    action_space = [0,1,2]

    states = []
    for c1 in range(11):
        for s1 in range(11):
            for c2 in range(11):
                for s2 in range(11):
                    for dot1 in range(11):
                        for dot2 in range(11):
                            states.append((c1,s1,c2,s2, dot1, dot2))
    Q = {}

    for state in states:
        for action in action_space:
            Q[state, action] = 0

    score = 0
    total_reward = np.zeros(n_games)

    for i  in range(n_games):
        obs = env.reset()
        done = False
        if i % 1000 == 0:
            print('episode', i, 'score', score, 'eps', eps)

        score = 0
        state = get_state(obs)
        action = max_action(Q, state) if np.random.random() > eps else env.action_space.sample()

        while not done:
            obs_, reward, done, info = env.step(action)

            state_ = get_state(obs_)

            action_ = max_action(Q, state) if np.random.random() > eps else env.action_space.sample()
            score += reward

            Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[state_, action_] - Q[state, action])

            state = state_
            action = action_
        total_reward[i] = score
        eps = eps - 2 / n_games if eps > 0.01 else 0.01

    plt.plot(total_reward)
    plt.show()