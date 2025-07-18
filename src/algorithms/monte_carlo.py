"""
Algorithmes Monte Carlo
Monte Carlo ES, On-policy first visit Monte Carlo Control, Off-policy Monte Carlo Control
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from .base_algorithm import BaseAlgorithm
from tqdm import tqdm


class MonteCarloBase(BaseAlgorithm):
    """
    Classe de base pour les algorithmes Monte Carlo.
    """
    
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # Hyperparamètres spécifiques
        self.epsilon = kwargs.get('epsilon', 0.1)  # Pour epsilon-greedy
        self.alpha = kwargs.get('alpha', None)  # Learning rate (si None, moyenne)
        
        # Structures pour Monte Carlo
        self.returns = defaultdict(list)  # G(s) ou G(s,a)
        self.visit_counts = defaultdict(int)  # N(s) ou N(s,a)
        
    def generate_episode(self, policy=None, exploring_starts=False, behavior_policy=None):
        """
        Génère un épisode complet.
        
        Args:
            policy: Politique à suivre (si None, utilise self.policy)
            exploring_starts: Si True, commence avec état et action aléatoires
            behavior_policy: Pour off-policy (si différent de policy)
            
        Returns:
            Liste de tuples (state, action, reward)
        """
        episode = []
        
        # Initialisation
        if exploring_starts:
            # État et action de départ aléatoires
            state = self.env.reset()
            actions = self.env.get_action_space()
            action = np.random.choice(actions)
            
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        else:
            state = self.env.reset()
            done = False
            
        # Générer l'épisode
        step = 0
        max_steps = 1000
        
        while not done and step < max_steps:
            # Choisir l'action
            if behavior_policy is not None:
                # Off-policy: utiliser behavior_policy
                action = self.select_action_from_policy(state, behavior_policy)
            elif policy is not None:
                action = self.select_action_from_policy(state, policy)
            else:
                # Utiliser la politique actuelle avec exploration
                action = self.select_action_epsilon_greedy(state)
                
            # Effectuer l'action
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            
            state = next_state
            step += 1
            
        return episode
        
    def select_action_epsilon_greedy(self, state):
        """
        Sélection d'action epsilon-greedy.
        """
        if np.random.random() < self.epsilon:
            # Exploration
            return np.random.choice(self.env.get_action_space())
        else:
            # Exploitation
            if state in self.policy:
                return self.policy[state]
            else:
                # État non vu, action aléatoire
                return np.random.choice(self.env.get_action_space())
                
    def select_action_from_policy(self, state, policy):
        """
        Sélectionne une action selon une politique donnée.
        """
        if isinstance(policy, dict):
            # Politique déterministe
            if state in policy:
                return policy[state]
            else:
                return np.random.choice(self.env.get_action_space())
        else:
            # Politique stochastique (fonction)
            return policy(state)
            
    def update_value_function(self):
        """
        Met à jour V(s) à partir de Q(s,a) et π.
        """
        for state in set(s for s, _ in self.q_function.keys()):
            if state in self.policy:
                action = self.policy[state]
                self.value_function[state] = self.q_function.get((state, action), 0)


class MonteCarloES(MonteCarloBase):
    """
    Monte Carlo with Exploring Starts.
    
    Garantit l'exploration en commençant chaque épisode avec un état
    et une action aléatoires.
    """
    
    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Monte Carlo ES.
        """
        if verbose:
            print("\n" + "="*50)
            print("MONTE CARLO ES (Exploring Starts)")
            print("="*50)
            
        # Initialisation
        # Politique arbitraire
        for state in set(s for s, _ in self.env.get_observation_space() 
                        if not isinstance(s, str) or s not in ['terminal', 'done']):
            self.policy[state] = np.random.choice(self.env.get_action_space())
            
        # Q(s,a) arbitraire
        for state in self.env.get_observation_space():
            for action in self.env.get_action_space():
                self.q_function[(state, action)] = 0.0
                
        # Boucle principale
        for episode_num in tqdm(range(self.episodes), desc="Entraînement MC-ES", disable=not verbose):
            # Générer un épisode avec exploring starts
            episode = self.generate_episode(exploring_starts=True)
            
            # Calculer les retours G
            G = 0
            visited_state_actions = set()
            
            # Parcourir l'épisode à l'envers
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                # First-visit MC
                if (state, action) not in visited_state_actions:
                    visited_state_actions.add((state, action))
                    
                    # Mise à jour de Q(s,a)
                    self.returns[(state, action)].append(G)
                    self.visit_counts[(state, action)] += 1
                    
                    if self.alpha is None:
                        # Moyenne des retours
                        self.q_function[(state, action)] = np.mean(self.returns[(state, action)])
                    else:
                        # Mise à jour incrémentale
                        self.q_function[(state, action)] += self.alpha * (G - self.q_function[(state, action)])
                        
            # Amélioration de la politique (greedy)
            for state in set(s for s, _ in episode):
                # Trouver la meilleure action
                action_values = {a: self.q_function.get((state, a), 0) 
                               for a in self.env.get_action_space()}
                best_action = max(action_values, key=action_values.get)
                self.policy[state] = best_action
                
            # Historique
            total_reward = sum(r for _, _, r in episode)
            self.training_history['episodes'].append(episode_num)
            self.training_history['rewards'].append(total_reward)
            self.training_history['steps'].append(len(episode))
            
        # Mise à jour finale de V(s)
        self.update_value_function()
        
        self.trained = True
        
        if verbose:
            print(f"\nEntraînement terminé!")
            print(f"Récompense moyenne (100 derniers): {np.mean(self.training_history['rewards'][-100:]):.2f}")
            
        return {
            'episodes_trained': self.episodes,
            'final_policy': self.policy.copy(),
            'final_q_function': self.q_function.copy()
        }
        
    def get_policy(self) -> Dict:
        return self.policy
        
    def get_value_function(self) -> Dict:
        return self.value_function
        
    def get_q_function(self) -> Dict:
        return self.q_function


class OnPolicyMonteCarlo(MonteCarloBase):
    """
    On-policy first-visit Monte Carlo Control.
    
    Utilise une politique epsilon-greedy pour garantir l'exploration.
    """
    
    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Entraîne l'algorithme On-policy Monte Carlo.
        """
        if verbose:
            print("\n" + "="*50)
            print("ON-POLICY MONTE CARLO CONTROL")
            print(f"Epsilon: {self.epsilon}")
            print("="*50)
            
        # Initialisation
        # Q(s,a) arbitraire
        for state in self.env.get_observation_space():
            for action in self.env.get_action_space():
                self.q_function[(state, action)] = 0.0
                
        # Politique epsilon-greedy initiale
        for state in self.env.get_observation_space():
            actions = self.env.get_action_space()
            if actions:
                self.policy[state] = np.random.choice(actions)
                
        # Boucle principale
        for episode_num in tqdm(range(self.episodes), desc="Entraînement On-Policy MC", disable=not verbose):
            # Générer un épisode avec la politique epsilon-greedy actuelle
            episode = self.generate_episode()
            
            # Calculer les retours G
            G = 0
            visited_state_actions = set()
            
            # Parcourir l'épisode à l'envers
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                # First-visit MC
                if (state, action) not in visited_state_actions:
                    visited_state_actions.add((state, action))
                    
                    # Mise à jour de Q(s,a)
                    self.returns[(state, action)].append(G)
                    self.visit_counts[(state, action)] += 1
                    
                    if self.alpha is None:
                        # Moyenne des retours
                        self.q_function[(state, action)] = np.mean(self.returns[(state, action)])
                    else:
                        # Mise à jour incrémentale
                        self.q_function[(state, action)] += self.alpha * (G - self.q_function[(state, action)])
                        
                    # Amélioration de la politique
                    # Trouver la meilleure action
                    action_values = {a: self.q_function.get((state, a), 0) 
                                   for a in self.env.get_action_space()}
                    best_action = max(action_values, key=action_values.get)
                    self.policy[state] = best_action
                    
            # Historique
            total_reward = sum(r for _, _, r in episode)
            self.training_history['episodes'].append(episode_num)
            self.training_history['rewards'].append(total_reward)
            self.training_history['steps'].append(len(episode))
            
        # Mise à jour finale de V(s)
        self.update_value_function()
        
        self.trained = True
        
        if verbose:
            print(f"\nEntraînement terminé!")
            print(f"Récompense moyenne (100 derniers): {np.mean(self.training_history['rewards'][-100:]):.2f}")
            
        return {
            'episodes_trained': self.episodes,
            'final_policy': self.policy.copy(),
            'final_q_function': self.q_function.copy(),
            'epsilon': self.epsilon
        }
        
    def get_policy(self) -> Dict:
        return self.policy
        
    def get_value_function(self) -> Dict:
        return self.value_function
        
    def get_q_function(self) -> Dict:
        return self.q_function


class OffPolicyMonteCarlo(MonteCarloBase):
    """
    Off-policy Monte Carlo Control.
    
    Apprend une politique optimale (target) en suivant une politique
    d'exploration (behavior).
    """
    
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # Politique comportementale (behavior) - toujours epsilon-greedy
        self.behavior_epsilon = kwargs.get('behavior_epsilon', 0.3)
        
        # Importance sampling
        self.C = defaultdict(float)  # Somme cumulée des poids d'importance
        
    def behavior_policy(self, state):
        """
        Politique comportementale (epsilon-greedy).
        """
        if np.random.random() < self.behavior_epsilon:
            return np.random.choice(self.env.get_action_space())
        else:
            # Suivre la politique target actuelle
            if state in self.policy:
                return self.policy[state]
            else:
                return np.random.choice(self.env.get_action_space())
                
    def get_action_probability(self, state, action, policy_type='behavior'):
        """
        Retourne la probabilité de choisir une action selon une politique.
        """
        actions = self.env.get_action_space()
        num_actions = len(actions)
        
        if policy_type == 'behavior':
            # Politique epsilon-greedy
            if state in self.policy and action == self.policy[state]:
                return 1 - self.behavior_epsilon + self.behavior_epsilon / num_actions
            else:
                return self.behavior_epsilon / num_actions
        else:  # target policy (deterministic)
            if state in self.policy and action == self.policy[state]:
                return 1.0
            else:
                return 0.0
                
    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Off-policy Monte Carlo.
        """
        if verbose:
            print("\n" + "="*50)
            print("OFF-POLICY MONTE CARLO CONTROL")
            print(f"Behavior epsilon: {self.behavior_epsilon}")
            print("="*50)
            
        # Initialisation
        # Q(s,a) arbitraire
        for state in self.env.get_observation_space():
            for action in self.env.get_action_space():
                self.q_function[(state, action)] = 0.0
                self.C[(state, action)] = 0.0
                
        # Politique target initiale (arbitraire)
        for state in self.env.get_observation_space():
            actions = self.env.get_action_space()
            if actions:
                self.policy[state] = np.random.choice(actions)
                
        # Boucle principale
        for episode_num in tqdm(range(self.episodes), desc="Entraînement Off-Policy MC", disable=not verbose):
            # Générer un épisode avec la politique comportementale
            episode = self.generate_episode(behavior_policy=self.behavior_policy)
            
            # Importance sampling ratio
            G = 0
            W = 1.0
            
            # Parcourir l'épisode à l'envers
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                
                # Mise à jour de C et Q
                self.C[(state, action)] += W
                self.q_function[(state, action)] += (W / self.C[(state, action)]) * \
                                                    (G - self.q_function[(state, action)])
                
                # Amélioration de la politique (greedy)
                action_values = {a: self.q_function.get((state, a), 0) 
                               for a in self.env.get_action_space()}
                best_action = max(action_values, key=action_values.get)
                self.policy[state] = best_action
                
                # Si l'action ne correspond pas à la politique target, arrêter
                if action != self.policy[state]:
                    break
                    
                # Mise à jour du ratio d'importance
                target_prob = self.get_action_probability(state, action, 'target')
                behavior_prob = self.get_action_probability(state, action, 'behavior')
                
                if behavior_prob > 0:
                    W *= target_prob / behavior_prob
                else:
                    break
                    
            # Historique
            total_reward = sum(r for _, _, r in episode)
            self.training_history['episodes'].append(episode_num)
            self.training_history['rewards'].append(total_reward)
            self.training_history['steps'].append(len(episode))
            
        # Mise à jour finale de V(s)
        self.update_value_function()
        
        self.trained = True
        
        if verbose:
            print(f"\nEntraînement terminé!")
            print(f"Récompense moyenne (100 derniers): {np.mean(self.training_history['rewards'][-100:]):.2f}")
            
        return {
            'episodes_trained': self.episodes,
            'final_policy': self.policy.copy(),
            'final_q_function': self.q_function.copy(),
            'behavior_epsilon': self.behavior_epsilon
        }
        
    def get_policy(self) -> Dict:
        return self.policy
        
    def get_value_function(self) -> Dict:
        return self.value_function
        
    def get_q_function(self) -> Dict:
        return self.q_function 