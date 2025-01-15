import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from collections import deque
import random
import os
import logging

from sklearn.exceptions import NotFittedError

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from interface import Agent

# Configure logging 
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def safe_predict(model, state):
    """
    Safely predicts using the model. Returns 0.0 if the model isn't fitted.
    """
    try:
        return model.predict(state.reshape(1, -1))[0]
    except NotFittedError:
        logging.debug("Model is not fitted yet. Returning 0.0 as default Q-value.")
        return 0.0

class ProjectAgent(Agent):
    def __init__(
        self,
        state_dim=6,
        action_dim=4,
        gamma=0.99,
        epsilon=1.0,             # Start with full exploration
        epsilon_min=0.1,         # Minimum exploration rate
        epsilon_decay=0.995,     # Decay rate for epsilon
        batch_size=64,
        n_iterations=20,         
        max_memory_size=200000,  
        model_save_path="fqi_models.joblib",
        exploration_episodes=5000  
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.max_memory_size = max_memory_size
        self.model_save_path = model_save_path
        self.exploration_episodes = exploration_episodes

        # Initialize models for each action with GradientBoostingRegressor
        self.models = [
            make_pipeline(
                StandardScaler(),
                GradientBoostingRegressor(
                    n_estimators=200,     
                    max_depth=10,         
                    learning_rate=0.1,    
                    random_state=42
                )
            )
            for _ in range(self.action_dim)
        ]

        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=self.max_memory_size)

    def act(self, observation, use_random=False, temperature=1.0):
        """
        Selects an action using Boltzmann Exploration.
        """
        if use_random or random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            logging.debug(f"Selected random action: {action}")
            return action
        else:
            q_values = []
            for idx, model in enumerate(self.models):
                try:
                    q = model.predict(observation.reshape(1, -1))[0]
                except NotFittedError:
                    q = 0.0
                    logging.debug(f"Model for action {idx} not fitted. Using Q-value: {q}")
                q_values.append(q)
            # Apply Boltzmann distribution
            exp_q = np.exp(np.array(q_values) / temperature)
            probabilities = exp_q / np.sum(exp_q)
            action = np.random.choice(self.action_dim, p=probabilities)
            logging.debug(f"Selected Boltzmann action: {action} with probabilities: {probabilities}")
            return action

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))
        logging.debug(f"Stored transition: Action {action}, Reward {reward}, Done {done}")

    def train(self):
        """
        Trains the Q-function approximators using the collected experience.
        Implements Double FQI to reduce overestimation bias.
        """
        if len(self.replay_buffer) < self.batch_size:
            logging.debug("Not enough data to train.")
            return  # Not enough data to train

        # Sample a batch of transitions
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])

        # Compute target Q-values using Double FQI
        target_q_values = np.zeros((self.batch_size, self.action_dim))
        for idx in range(self.batch_size):
            if dones[idx]:
                target_q_values[idx] = rewards[idx]
            else:
                # Select a random action for target estimation to reduce bias
                random_action = random.randint(0, self.action_dim - 1)
                try:
                    target_q = self.models[random_action].predict(next_states[idx].reshape(1, -1))[0]
                except NotFittedError:
                    target_q = 0.0
                    logging.debug(f"Model for action {random_action} not fitted. Using Q-value: {target_q}")
                target_q_values[idx] = rewards[idx] + self.gamma * target_q

        # Update models for each action
        for action in range(self.action_dim):
            # Select samples where the action was taken
            mask = actions == action
            if np.sum(mask) == 0:
                continue  # No samples for this action in the batch
            X_train = states[mask]
            y_train = target_q_values[mask, action]
            self.models[action].fit(X_train, y_train)
            logging.debug(f"Trained model for action {action} with {len(X_train)} samples.")

    def save(self, path=None):
        """
        Saves the trained models to disk.
        """
        if path is None:
            path = self.model_save_path
        joblib.dump(self.models, path)
        logging.info(f"Models saved to {path}")

    def load(self, path=None):
        """
        Loads the trained models from disk.
        """
        if path is None:
            path = self.model_save_path
        if os.path.exists(path):
            self.models = joblib.load(path)
            logging.info(f"Models loaded from {path}")
        else:
            logging.warning(f"No model found at {path}. Please train the agent first.")

    def explore_policy(self, env, n_episodes=5000, temperature=1.0):
        """
        Collects experience by interacting with the environment using an exploration policy.
        Utilizes Boltzmann Exploration.
        """
        for episode in range(n_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            step_count = 0
            if (episode + 1) % 500 == 0 or episode == 0:
                logging.info(f"Starting Exploration Episode {episode + 1}/{n_episodes}")
            while not done and not truncated:
                action = self.act(state, use_random=True, temperature=temperature)  # Boltzmann Exploration
                next_state, reward, done, truncated, _ = env.step(action)
                self.store_transition(state, action, reward, next_state, done)
                state = next_state
                step_count += 1
            if (episode + 1) % 50 == 0 or episode == n_episodes - 1:
                logging.info(f"Completed Exploration Episode {episode + 1}/{n_episodes} in {step_count} steps.")
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)
                logging.debug(f"Epsilon decayed to {self.epsilon}")

    def train_fqi(self, env):
        """
        Executes the FQI training process.
        """
        # Collect initial experience
        logging.info("Collecting initial experience...")
        self.explore_policy(env=env, n_episodes=self.exploration_episodes)

        # Iteratively train the Q-function approximators
        logging.info("Starting FQI iterations...")
        for iteration in range(self.n_iterations):
            logging.info(f"FQI Iteration {iteration + 1}/{self.n_iterations} - Training...")
            self.train()
            logging.info(f"FQI Iteration {iteration + 1}/{self.n_iterations} - Training Completed.")

        # Save the final model
        self.save()  # Saves as "fqi_models.joblib"
        logging.info("Final model saved as fqi_models.joblib")

        logging.info("FQI training completed.")


def main():
    # Instantiate the environment with TimeLimit wrapper
    env = TimeLimit(
        env=HIVPatient(domain_randomization=False),
        max_episode_steps=200
    )

    # Initialize the agent 
    agent = ProjectAgent(
        exploration_episodes=200,  
        n_iterations=20,            
        max_memory_size=20000,     
    )

    # Train the agent
    agent.train_fqi(env=env)

    # Save the trained models
    agent.save()

if __name__ == "__main__":
    main()
