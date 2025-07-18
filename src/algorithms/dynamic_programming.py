"""
Algorithmes de programmation dynamique
Policy Iteration et Value Iteration
"""

import numpy as np
from typing import Dict, Any, Optional
from collections import defaultdict
from .base_algorithm import BaseAlgorithm
from tqdm import tqdm


class DynamicProgrammingBase(BaseAlgorithm):
    """
    Classe de base pour les algorithmes de programmation dynamique.
    Nécessite un modèle de l'environnement (probabilités de transition).
    """
    
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        
        # Hyperparamètres spécifiques
        self.theta = kwargs.get('theta', 1e-6)  # Seuil de convergence
        self.max_iterations = kwargs.get('max_iterations', 1000)
        
        # Construire le modèle si nécessaire
        self.build_model()
        
    def build_model(self):
        """
        Construit le modèle de l'environnement en échantillonnant.
        Pour les environnements déterministes simples.
        """
        self.states = []
        self.actions = self.env.get_action_space()
        
        # Pour les environnements simples, on peut énumérer les états
        if hasattr(self.env, 'get_observation_space'):
            self.states = self.env.get_observation_space()
        else:
            # Sinon, découvrir les états par exploration
            print("Découverte des états par exploration...")
            discovered_states = set()
            
            for _ in range(100):  # Plusieurs épisodes d'exploration
                state = self.env.reset()
                discovered_states.add(str(state))
                
                for _ in range(100):  # Limite de pas par épisode
                    action = np.random.choice(self.actions)
                    next_state, _, done, _ = self.env.step(action)
                    discovered_states.add(str(next_state))
                    
                    if done:
                        break
                        
                    state = next_state
                    
            self.states = list(discovered_states)
            
        # Initialiser les structures
        self.initialize_values()
        
    def initialize_values(self):
        """Initialise les fonctions de valeur et la politique."""
        # Initialiser V(s) et π(s)
        for state in self.states:
            self.value_function[state] = 0.0
            self.policy[state] = np.random.choice(self.actions)
            
            # Initialiser Q(s,a)
            for action in self.actions:
                self.q_function[(state, action)] = 0.0
                
    def get_transition_probability(self, state, action, next_state, reward):
        """
        Retourne P(s',r|s,a).
        Pour les environnements déterministes, c'est 0 ou 1.
        """
        # Sauvegarder l'état actuel
        current_state = self.env.state
        current_done = self.env.done
        
        # Tester la transition
        self.env.state = state
        self.env.done = False
        
        actual_next_state, actual_reward, _, _ = self.env.step(action)
        
        # Restaurer l'état
        self.env.state = current_state
        self.env.done = current_done
        
        # Vérifier si la transition correspond
        if str(actual_next_state) == str(next_state) and abs(actual_reward - reward) < 1e-6:
            return 1.0
        else:
            return 0.0
            
    def get_expected_return(self, state, action):
        """
        Calcule le retour espéré pour une paire (état, action).
        """
        total_return = 0.0
        
        # Sauvegarder l'état actuel
        current_state = self.env.state
        current_done = self.env.done
        
        # Simuler l'action
        self.env.state = state
        self.env.done = False
        
        next_state, reward, done, _ = self.env.step(action)
        
        # Restaurer l'état
        self.env.state = current_state
        self.env.done = current_done
        
        # Calculer le retour
        if done:
            total_return = reward
        else:
            total_return = reward + self.gamma * self.value_function.get(next_state, 0)
            
        return total_return


class PolicyIteration(DynamicProgrammingBase):
    """
    Algorithme Policy Iteration.
    
    Alterne entre:
    1. Policy Evaluation: évaluer V^π pour la politique actuelle
    2. Policy Improvement: améliorer la politique en étant greedy par rapport à V
    """
    
    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Entraîne l'algorithme par Policy Iteration.
        """
        if verbose:
            print("\n" + "="*50)
            print("POLICY ITERATION")
            print("="*50)
            
        iterations = 0
        policy_stable = False
        
        while not policy_stable and iterations < self.max_iterations:
            iterations += 1
            
            if verbose:
                print(f"\nItération {iterations}")
                
            # Policy Evaluation
            self.policy_evaluation(verbose=verbose and iterations <= 3)
            
            # Policy Improvement
            policy_stable = self.policy_improvement(verbose=verbose and iterations <= 3)
            
            if verbose and iterations <= 3:
                print(f"Politique stable: {policy_stable}")
                
        if verbose:
            print(f"\nConvergence atteinte en {iterations} itérations")
            
        self.trained = True
        
        # Calculer Q-values finales
        self.compute_q_values()
        
        return {
            'iterations': iterations,
            'converged': policy_stable,
            'final_value_function': self.value_function.copy(),
            'final_policy': self.policy.copy()
        }
        
    def policy_evaluation(self, verbose: bool = False):
        """
        Évalue la politique actuelle (calcule V^π).
        """
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            delta = 0
            
            # Pour chaque état
            for state in self.states:
                if isinstance(state, str) and state in ['terminal', 'done']:
                    continue
                    
                old_value = self.value_function[state]
                
                # Calculer la nouvelle valeur selon la politique actuelle
                action = self.policy[state]
                self.value_function[state] = self.get_expected_return(state, action)
                
                delta = max(delta, abs(old_value - self.value_function[state]))
                
            if verbose:
                print(f"  Evaluation iteration {iteration}, delta = {delta:.6f}")
                
            if delta < self.theta:
                break
                
    def policy_improvement(self, verbose: bool = False) -> bool:
        """
        Améliore la politique en étant greedy par rapport à V.
        
        Returns:
            True si la politique est stable (pas de changement)
        """
        policy_stable = True
        
        for state in self.states:
            if isinstance(state, str) and state in ['terminal', 'done']:
                continue
                
            old_action = self.policy[state]
            
            # Trouver la meilleure action
            action_values = {}
            for action in self.actions:
                action_values[action] = self.get_expected_return(state, action)
                
            # Choisir l'action avec la valeur maximale
            best_action = max(action_values, key=action_values.get)
            self.policy[state] = best_action
            
            if old_action != best_action:
                policy_stable = False
                
            if verbose:
                print(f"  État {state}: {old_action} -> {best_action}")
                
        return policy_stable
        
    def compute_q_values(self):
        """Calcule les Q-values à partir de V et π."""
        for state in self.states:
            for action in self.actions:
                self.q_function[(state, action)] = self.get_expected_return(state, action)
                
    def get_policy(self) -> Dict:
        return self.policy
        
    def get_value_function(self) -> Dict:
        return self.value_function
        
    def get_q_function(self) -> Dict:
        return self.q_function


class ValueIteration(DynamicProgrammingBase):
    """
    Algorithme Value Iteration.
    
    Calcule directement V* en utilisant l'équation d'optimalité de Bellman.
    La politique optimale est ensuite extraite de V*.
    """
    
    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Entraîne l'algorithme par Value Iteration.
        """
        if verbose:
            print("\n" + "="*50)
            print("VALUE ITERATION")
            print("="*50)
            
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            delta = 0
            
            # Pour chaque état
            for state in self.states:
                if isinstance(state, str) and state in ['terminal', 'done']:
                    continue
                    
                old_value = self.value_function[state]
                
                # Calculer la valeur maximale sur toutes les actions
                action_values = []
                for action in self.actions:
                    action_values.append(self.get_expected_return(state, action))
                    
                self.value_function[state] = max(action_values) if action_values else 0
                
                delta = max(delta, abs(old_value - self.value_function[state]))
                
            if verbose and (iteration <= 10 or iteration % 10 == 0):
                print(f"Itération {iteration}, delta = {delta:.6f}")
                
            # Vérifier la convergence
            if delta < self.theta:
                if verbose:
                    print(f"\nConvergence atteinte en {iteration} itérations")
                break
                
        # Extraire la politique optimale
        self.extract_policy(verbose=verbose)
        
        # Calculer les Q-values
        self.compute_q_values()
        
        self.trained = True
        
        return {
            'iterations': iteration,
            'converged': delta < self.theta,
            'final_value_function': self.value_function.copy(),
            'final_policy': self.policy.copy()
        }
        
    def extract_policy(self, verbose: bool = False):
        """
        Extrait la politique optimale à partir de V*.
        """
        if verbose:
            print("\nExtraction de la politique optimale...")
            
        for state in self.states:
            if isinstance(state, str) and state in ['terminal', 'done']:
                continue
                
            # Trouver la meilleure action
            action_values = {}
            for action in self.actions:
                action_values[action] = self.get_expected_return(state, action)
                
            # Choisir l'action avec la valeur maximale
            if action_values:
                best_action = max(action_values, key=action_values.get)
                self.policy[state] = best_action
                
                if verbose and len(self.states) <= 20:  # Afficher seulement pour petits environnements
                    print(f"  État {state}: action optimale = {best_action}")
                    
    def compute_q_values(self):
        """Calcule les Q-values à partir de V* et π*."""
        for state in self.states:
            for action in self.actions:
                self.q_function[(state, action)] = self.get_expected_return(state, action)
                
    def get_policy(self) -> Dict:
        return self.policy
        
    def get_value_function(self) -> Dict:
        return self.value_function
        
    def get_q_function(self) -> Dict:
        return self.q_function 