"""
Algorithmes de Planning
Dyna-Q et Dyna-Q+
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from collections import defaultdict
from .base_algorithm import BaseAlgorithm
from tqdm import tqdm
import time


class Model:
    """
    Modèle de l'environnement pour la planification.
    """
    
    def __init__(self):
        self.transitions = {}  # (state, action) -> (next_state, reward)
        self.state_action_pairs = []  # Liste des paires (état, action) vues
        
    def update(self, state, action, next_state, reward):
        """
        Met à jour le modèle avec une nouvelle transition.
        """
        self.transitions[(state, action)] = (next_state, reward)
        
        # Ajouter à la liste si nouveau
        if (state, action) not in self.state_action_pairs:
            self.state_action_pairs.append((state, action))
            
    def sample(self, state, action) -> Tuple[Any, float]:
        """
        Échantillonne une transition du modèle.
        """
        if (state, action) in self.transitions:
            return self.transitions[(state, action)]
        else:
            # Transition non vue, retourner état actuel et récompense 0
            return state, 0.0
            
    def get_random_state_action(self) -> Optional[Tuple[Any, Any]]:
        """
        Retourne une paire (état, action) aléatoire du modèle.
        """
        if self.state_action_pairs:
            return self.state_action_pairs[np.random.randint(len(self.state_action_pairs))]
        return None


class DynaQ(BaseAlgorithm):
    """
    Algorithme Dyna-Q.
    
    Combine l'apprentissage direct (Q-Learning) avec la planification
    en utilisant un modèle appris de l'environnement.
    """
    
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # Hyperparamètres spécifiques
        self.alpha = kwargs.get('alpha', 0.1)  # Learning rate
        self.epsilon = kwargs.get('epsilon', 0.1)  # Pour epsilon-greedy
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        self.epsilon_min = kwargs.get('epsilon_min', 0.01)
        self.planning_steps = kwargs.get('planning_steps', 5)  # Nombre de pas de planification
        
        # Initialiser Q(s,a) et le modèle
        def zero_value():
            return 0.0
        self.q_function = defaultdict(zero_value)
        self.model = Model()
        
    def select_action_epsilon_greedy(self, state, epsilon=None):
        """
        Sélection d'action epsilon-greedy.
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        if np.random.random() < epsilon:
            return np.random.choice(self.env.get_action_space())
        else:
            return self.get_greedy_action(state)
            
    def get_greedy_action(self, state):
        """
        Retourne l'action greedy pour un état donné.
        """
        actions = self.env.get_action_space()
        if not actions:
            return None
            
        q_values = {action: self.q_function[(state, action)] for action in actions}
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        
        return np.random.choice(best_actions)
        
    def q_learning_update(self, state, action, reward, next_state, done):
        """
        Mise à jour Q-Learning.
        """
        current_q = self.q_function[(state, action)]
        
        if done:
            target = reward
        else:
            next_q_values = [self.q_function[(next_state, a)] 
                           for a in self.env.get_action_space()]
            max_next_q = max(next_q_values) if next_q_values else 0
            target = reward + self.gamma * max_next_q
            
        self.q_function[(state, action)] = current_q + self.alpha * (target - current_q)
        
    def planning_step(self):
        """
        Effectue un pas de planification en utilisant le modèle.
        """
        # Échantillonner une paire (état, action) du modèle
        state_action = self.model.get_random_state_action()
        if state_action is None:
            return
            
        state, action = state_action
        
        # Simuler la transition avec le modèle
        next_state, reward = self.model.sample(state, action)
        
        # Vérifier si c'est un état terminal
        # (approximation: si la récompense est non nulle dans les environnements simples)
        done = abs(reward) > 0.5  # Adapter selon l'environnement
        
        # Mise à jour Q-Learning sur la transition simulée
        self.q_learning_update(state, action, reward, next_state, done)
        
    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Dyna-Q.
        """
        if verbose:
            print("\n" + "="*50)
            print("DYNA-Q")
            print(f"Alpha: {self.alpha}, Epsilon: {self.epsilon}")
            print(f"Planning steps: {self.planning_steps}")
            print("="*50)
            
        current_epsilon = self.epsilon
        total_planning_steps = 0
        
        # Boucle sur les épisodes
        for episode_num in tqdm(range(self.episodes), desc="Entraînement Dyna-Q", disable=not verbose):
            state = self.env.reset()
            
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < 1000:
                # (a) Sélectionner une action
                action = self.select_action_epsilon_greedy(state, current_epsilon)
                
                # (b) Exécuter l'action
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                # (c) Apprentissage direct (Q-Learning)
                self.q_learning_update(state, action, reward, next_state, done)
                
                # (d) Mise à jour du modèle
                self.model.update(state, action, next_state, reward)
                
                # (e) Planification
                for _ in range(self.planning_steps):
                    self.planning_step()
                    total_planning_steps += 1
                    
                state = next_state
                
            # Historique
            self.training_history['episodes'].append(episode_num)
            self.training_history['rewards'].append(episode_reward)
            self.training_history['steps'].append(episode_steps)
            
            # Décroissance d'epsilon
            current_epsilon = max(self.epsilon_min, current_epsilon * self.epsilon_decay)
            
        # Mise à jour finale de la politique
        self.update_policy()
        self.update_value_function()
        
        self.trained = True
        
        if verbose:
            print(f"\nEntraînement terminé!")
            print(f"Epsilon final: {current_epsilon:.4f}")
            print(f"Total planning steps: {total_planning_steps}")
            print(f"Récompense moyenne (100 derniers): {np.mean(self.training_history['rewards'][-100:]):.2f}")
            
        return {
            'episodes_trained': self.episodes,
            'final_policy': self.policy.copy(),
            'final_q_function': dict(self.q_function),
            'total_planning_steps': total_planning_steps,
            'model_size': len(self.model.state_action_pairs)
        }
        
    def update_policy(self):
        """Met à jour la politique."""
        visited_states = set(s for s, _ in self.q_function.keys())
        for state in visited_states:
            self.policy[state] = self.get_greedy_action(state)
            
    def update_value_function(self):
        """Met à jour V(s)."""
        visited_states = set(s for s, _ in self.q_function.keys())
        for state in visited_states:
            if state in self.policy:
                action = self.policy[state]
                self.value_function[state] = self.q_function[(state, action)]
                
    def get_policy(self) -> Dict:
        return self.policy
        
    def get_value_function(self) -> Dict:
        return self.value_function
        
    def get_q_function(self) -> Dict:
        return dict(self.q_function)


class DynaQPlus(DynaQ):
    """
    Algorithme Dyna-Q+.
    
    Extension de Dyna-Q qui encourage l'exploration des paires (état, action)
    non visitées depuis longtemps.
    """
    
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # Hyperparamètre supplémentaire
        self.kappa = kwargs.get('kappa', 0.001)  # Bonus d'exploration
        
        # Temps depuis la dernière visite
        self.time_since_visit = defaultdict(int)
        self.current_time = 0
        
    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Entraîne l'algorithme Dyna-Q+.
        """
        if verbose:
            print("\n" + "="*50)
            print("DYNA-Q+")
            print(f"Alpha: {self.alpha}, Epsilon: {self.epsilon}")
            print(f"Planning steps: {self.planning_steps}")
            print(f"Kappa (exploration bonus): {self.kappa}")
            print("="*50)
            
        current_epsilon = self.epsilon
        total_planning_steps = 0
        
        # Boucle sur les épisodes
        for episode_num in tqdm(range(self.episodes), desc="Entraînement Dyna-Q+", disable=not verbose):
            state = self.env.reset()
            
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < 1000:
                # Incrémenter le temps
                self.current_time += 1
                
                # (a) Sélectionner une action
                action = self.select_action_epsilon_greedy(state, current_epsilon)
                
                # (b) Exécuter l'action
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                # Mettre à jour le temps de visite
                self.time_since_visit[(state, action)] = self.current_time
                
                # (c) Apprentissage direct (Q-Learning avec bonus)
                self.q_learning_update_with_bonus(state, action, reward, next_state, done)
                
                # (d) Mise à jour du modèle
                self.model.update(state, action, next_state, reward)
                
                # (e) Planification avec bonus d'exploration
                for _ in range(self.planning_steps):
                    self.planning_step_with_bonus()
                    total_planning_steps += 1
                    
                state = next_state
                
            # Historique
            self.training_history['episodes'].append(episode_num)
            self.training_history['rewards'].append(episode_reward)
            self.training_history['steps'].append(episode_steps)
            
            # Décroissance d'epsilon
            current_epsilon = max(self.epsilon_min, current_epsilon * self.epsilon_decay)
            
        # Mise à jour finale
        self.update_policy()
        self.update_value_function()
        
        self.trained = True
        
        if verbose:
            print(f"\nEntraînement terminé!")
            print(f"Epsilon final: {current_epsilon:.4f}")
            print(f"Total planning steps: {total_planning_steps}")
            print(f"Récompense moyenne (100 derniers): {np.mean(self.training_history['rewards'][-100:]):.2f}")
            
        return {
            'episodes_trained': self.episodes,
            'final_policy': self.policy.copy(),
            'final_q_function': dict(self.q_function),
            'total_planning_steps': total_planning_steps,
            'model_size': len(self.model.state_action_pairs),
            'kappa': self.kappa
        }
        
    def q_learning_update_with_bonus(self, state, action, reward, next_state, done):
        """
        Mise à jour Q-Learning avec bonus d'exploration.
        """
        current_q = self.q_function[(state, action)]
        
        # Ajouter le bonus d'exploration
        time_diff = self.current_time - self.time_since_visit.get((state, action), 0)
        bonus = self.kappa * np.sqrt(time_diff)
        reward_with_bonus = reward + bonus
        
        if done:
            target = reward_with_bonus
        else:
            next_q_values = [self.q_function[(next_state, a)] 
                           for a in self.env.get_action_space()]
            max_next_q = max(next_q_values) if next_q_values else 0
            target = reward_with_bonus + self.gamma * max_next_q
            
        self.q_function[(state, action)] = current_q + self.alpha * (target - current_q)
        
    def planning_step_with_bonus(self):
        """
        Pas de planification avec bonus d'exploration.
        """
        state_action = self.model.get_random_state_action()
        if state_action is None:
            return
            
        state, action = state_action
        next_state, reward = self.model.sample(state, action)
        
        # Ajouter le bonus d'exploration pour la planification
        time_diff = self.current_time - self.time_since_visit.get((state, action), 0)
        bonus = self.kappa * np.sqrt(time_diff)
        
        # Vérifier si terminal
        done = abs(reward) > 0.5
        
        # Mise à jour avec bonus
        self.q_learning_update_with_bonus(state, action, reward, next_state, done) 