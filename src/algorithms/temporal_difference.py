"""
Algorithmes Temporal Difference Learning
Sarsa, Q-Learning, Expected Sarsa
"""

import numpy as np
from typing import Dict, Any, Optional
from collections import defaultdict
from .base_algorithm import BaseAlgorithm
from tqdm import tqdm


class TDBase(BaseAlgorithm):
    """
    Classe de base pour les algorithmes TD.
    """
    
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # Hyperparamètres spécifiques TD
        self.alpha = kwargs.get('alpha', 0.1)  # Learning rate
        self.epsilon = kwargs.get('epsilon', 0.1)  # Pour epsilon-greedy
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)  # Décroissance d'epsilon
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)  # Epsilon minimum
        
        # Initialiser Q(s,a)
        self.q_function = defaultdict(lambda: 0.0)
        
    def select_action_epsilon_greedy(self, state, epsilon=None):
        """
        Sélection d'action epsilon-greedy.
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() < epsilon:
            # Exploration
            return np.random.choice(self.env.get_action_space())
        else:
            # Exploitation
            return self.get_greedy_action(state)
            
    def get_greedy_action(self, state):
        """
        Retourne l'action greedy pour un état donné.
        """
        actions = self.env.get_action_space()
        if not actions:
            return None
            
        # Obtenir les Q-values pour toutes les actions
        q_values = {action: self.q_function[(state, action)] for action in actions}
        
        # Retourner l'action avec la Q-value maximale
        # En cas d'égalité, choisir aléatoirement
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        
        return np.random.choice(best_actions)
        
    def update_policy(self):
        """
        Met à jour la politique en étant greedy par rapport à Q.
        """
        visited_states = set(s for s, _ in self.q_function.keys())
        
        for state in visited_states:
            self.policy[state] = self.get_greedy_action(state)
            
    def update_value_function(self):
        """
        Met à jour V(s) à partir de Q(s,a) et π.
        """
        visited_states = set(s for s, _ in self.q_function.keys())
        
        for state in visited_states:
            if state in self.policy:
                action = self.policy[state]
                self.value_function[state] = self.q_function[(state, action)]


class Sarsa(TDBase):
    """
    Algorithme SARSA (State-Action-Reward-State-Action).
    
    Algorithme on-policy qui apprend Q(s,a) en suivant une politique epsilon-greedy.
    """
    
    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Entraîne l'algorithme SARSA.
        """
        if verbose:
            print("\n" + "="*50)
            print("SARSA")
            print(f"Alpha: {self.alpha}, Epsilon: {self.epsilon}")
            print("="*50)
            
        # Variable pour l'epsilon décroissant
        current_epsilon = self.epsilon
        
        # Boucle sur les épisodes
        for episode_num in tqdm(range(self.episodes), desc="Entraînement SARSA", disable=not verbose):
            # Initialiser S
            state = self.env.reset()
            
            # Choisir A depuis S en utilisant la politique (epsilon-greedy)
            action = self.select_action_epsilon_greedy(state, current_epsilon)
            
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # Boucle sur les pas de l'épisode
            while not done and episode_steps < 1000:
                # Prendre l'action A, observer R, S'
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                # Choisir A' depuis S' en utilisant la politique
                if not done:
                    next_action = self.select_action_epsilon_greedy(next_state, current_epsilon)
                else:
                    next_action = None
                    
                # Mise à jour SARSA
                current_q = self.q_function[(state, action)]
                
                if done:
                    # État terminal
                    target = reward
                else:
                    # Q(S', A')
                    next_q = self.q_function[(next_state, next_action)]
                    target = reward + self.gamma * next_q
                    
                # Mise à jour Q(S, A)
                self.q_function[(state, action)] = current_q + self.alpha * (target - current_q)
                
                # S ← S', A ← A'
                state = next_state
                action = next_action
                
            # Historique
            self.training_history['episodes'].append(episode_num)
            self.training_history['rewards'].append(episode_reward)
            self.training_history['steps'].append(episode_steps)
            
            # Décroissance d'epsilon
            current_epsilon = max(self.epsilon_min, current_epsilon * self.epsilon_decay)
            
        # Mise à jour finale de la politique et de V(s)
        self.update_policy()
        self.update_value_function()
        
        self.trained = True
        
        if verbose:
            print(f"\nEntraînement terminé!")
            print(f"Epsilon final: {current_epsilon:.4f}")
            print(f"Récompense moyenne (100 derniers): {np.mean(self.training_history['rewards'][-100:]):.2f}")
            
        return {
            'episodes_trained': self.episodes,
            'final_policy': self.policy.copy(),
            'final_q_function': dict(self.q_function),
            'final_epsilon': current_epsilon
        }
        
    def get_policy(self) -> Dict:
        return self.policy
        
    def get_value_function(self) -> Dict:
        return self.value_function
        
    def get_q_function(self) -> Dict:
        return dict(self.q_function)


class QLearning(TDBase):
    """
    Algorithme Q-Learning.
    
    Algorithme off-policy qui apprend Q*(s,a) directement.
    """
    
    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Q-Learning.
        """
        if verbose:
            print("\n" + "="*50)
            print("Q-LEARNING")
            print(f"Alpha: {self.alpha}, Epsilon: {self.epsilon}")
            print("="*50)
            
        # Variable pour l'epsilon décroissant
        current_epsilon = self.epsilon
        
        # Boucle sur les épisodes
        for episode_num in tqdm(range(self.episodes), desc="Entraînement Q-Learning", disable=not verbose):
            # Initialiser S
            state = self.env.reset()
            
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # Boucle sur les pas de l'épisode
            while not done and episode_steps < 1000:
                # Choisir A depuis S en utilisant la politique dérivée de Q (epsilon-greedy)
                action = self.select_action_epsilon_greedy(state, current_epsilon)
                
                # Prendre l'action A, observer R, S'
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                # Mise à jour Q-Learning
                current_q = self.q_function[(state, action)]
                
                if done:
                    # État terminal
                    target = reward
                else:
                    # max_a Q(S', a)
                    next_q_values = [self.q_function[(next_state, a)] 
                                   for a in self.env.get_action_space()]
                    max_next_q = max(next_q_values) if next_q_values else 0
                    target = reward + self.gamma * max_next_q
                    
                # Mise à jour Q(S, A)
                self.q_function[(state, action)] = current_q + self.alpha * (target - current_q)
                
                # S ← S'
                state = next_state
                
            # Historique
            self.training_history['episodes'].append(episode_num)
            self.training_history['rewards'].append(episode_reward)
            self.training_history['steps'].append(episode_steps)
            
            # Décroissance d'epsilon
            current_epsilon = max(self.epsilon_min, current_epsilon * self.epsilon_decay)
            
        # Mise à jour finale de la politique et de V(s)
        self.update_policy()
        self.update_value_function()
        
        self.trained = True
        
        if verbose:
            print(f"\nEntraînement terminé!")
            print(f"Epsilon final: {current_epsilon:.4f}")
            print(f"Récompense moyenne (100 derniers): {np.mean(self.training_history['rewards'][-100:]):.2f}")
            
        return {
            'episodes_trained': self.episodes,
            'final_policy': self.policy.copy(),
            'final_q_function': dict(self.q_function),
            'final_epsilon': current_epsilon
        }
        
    def get_policy(self) -> Dict:
        return self.policy
        
    def get_value_function(self) -> Dict:
        return self.value_function
        
    def get_q_function(self) -> Dict:
        return dict(self.q_function)


class ExpectedSarsa(TDBase):
    """
    Algorithme Expected SARSA.
    
    Variante de SARSA qui utilise la valeur espérée sous la politique actuelle
    au lieu d'échantillonner une action spécifique.
    """
    
    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Expected SARSA.
        """
        if verbose:
            print("\n" + "="*50)
            print("EXPECTED SARSA")
            print(f"Alpha: {self.alpha}, Epsilon: {self.epsilon}")
            print("="*50)
            
        # Variable pour l'epsilon décroissant
        current_epsilon = self.epsilon
        
        # Boucle sur les épisodes
        for episode_num in tqdm(range(self.episodes), desc="Entraînement Expected SARSA", disable=not verbose):
            # Initialiser S
            state = self.env.reset()
            
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # Boucle sur les pas de l'épisode
            while not done and episode_steps < 1000:
                # Choisir A depuis S en utilisant la politique (epsilon-greedy)
                action = self.select_action_epsilon_greedy(state, current_epsilon)
                
                # Prendre l'action A, observer R, S'
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                # Mise à jour Expected SARSA
                current_q = self.q_function[(state, action)]
                
                if done:
                    # État terminal
                    target = reward
                else:
                    # Calculer l'espérance sur les actions possibles
                    expected_value = 0.0
                    actions = self.env.get_action_space()
                    num_actions = len(actions)
                    
                    # Action greedy
                    greedy_action = self.get_greedy_action(next_state)
                    
                    for a in actions:
                        # Probabilité de choisir cette action avec epsilon-greedy
                        if a == greedy_action:
                            prob = (1 - current_epsilon) + current_epsilon / num_actions
                        else:
                            prob = current_epsilon / num_actions
                            
                        expected_value += prob * self.q_function[(next_state, a)]
                        
                    target = reward + self.gamma * expected_value
                    
                # Mise à jour Q(S, A)
                self.q_function[(state, action)] = current_q + self.alpha * (target - current_q)
                
                # S ← S'
                state = next_state
                
            # Historique
            self.training_history['episodes'].append(episode_num)
            self.training_history['rewards'].append(episode_reward)
            self.training_history['steps'].append(episode_steps)
            
            # Décroissance d'epsilon
            current_epsilon = max(self.epsilon_min, current_epsilon * self.epsilon_decay)
            
        # Mise à jour finale de la politique et de V(s)
        self.update_policy()
        self.update_value_function()
        
        self.trained = True
        
        if verbose:
            print(f"\nEntraînement terminé!")
            print(f"Epsilon final: {current_epsilon:.4f}")
            print(f"Récompense moyenne (100 derniers): {np.mean(self.training_history['rewards'][-100:]):.2f}")
            
        return {
            'episodes_trained': self.episodes,
            'final_policy': self.policy.copy(),
            'final_q_function': dict(self.q_function),
            'final_epsilon': current_epsilon
        }
        
    def get_policy(self) -> Dict:
        return self.policy
        
    def get_value_function(self) -> Dict:
        return self.value_function
        
    def get_q_function(self) -> Dict:
        return dict(self.q_function) 