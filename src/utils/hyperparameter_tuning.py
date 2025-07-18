"""
Optimisation des hyperparamètres
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Type, Callable
import itertools
from tqdm import tqdm

from ..environments import BaseEnvironment
from ..algorithms import BaseAlgorithm


class HyperparameterTuner:
    """
    Classe pour l'optimisation des hyperparamètres.
    """
    
    def __init__(self):
        self.results = []
        
    def grid_search(self,
                   algorithm_class: Type[BaseAlgorithm],
                   env_class: Type[BaseEnvironment],
                   param_grid: Dict[str, List[Any]],
                   evaluation_metric: str = 'mean_reward',
                   num_trials: int = 1,
                   num_episodes: int = 1000,
                   num_eval_episodes: int = 100,
                   env_params: Dict[str, Any] = None,
                   verbose: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Recherche en grille des hyperparamètres.
        
        Args:
            algorithm_class: Classe de l'algorithme
            env_class: Classe de l'environnement
            param_grid: Grille des hyperparamètres à tester
            evaluation_metric: Métrique pour l'évaluation
            num_trials: Nombre d'essais par configuration
            num_episodes: Nombre d'épisodes d'entraînement
            num_eval_episodes: Nombre d'épisodes d'évaluation
            env_params: Paramètres de l'environnement
            verbose: Affichage détaillé
            
        Returns:
            best_params: Meilleurs hyperparamètres trouvés
            all_results: Tous les résultats
        """
        if env_params is None:
            env_params = {}
            
        # Générer toutes les combinaisons
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        print(f"\nRecherche en grille des hyperparamètres")
        print(f"Algorithme: {algorithm_class.__name__}")
        print(f"Environnement: {env_class.__name__}")
        print(f"Nombre de combinaisons: {len(param_combinations)}")
        print(f"Nombre d'essais par combinaison: {num_trials}")
        print(f"Total d'expérimentations: {len(param_combinations) * num_trials}")
        
        all_results = []
        best_score = -np.inf
        best_params = None
        
        # Tester chaque combinaison
        for params_tuple in tqdm(param_combinations, desc="Combinaisons"):
            # Créer le dictionnaire de paramètres
            params = dict(zip(param_names, params_tuple))
            params['episodes'] = num_episodes
            
            if verbose:
                print(f"\nTest des paramètres: {params}")
                
            # Plusieurs essais pour chaque configuration
            trial_scores = []
            
            for trial in range(num_trials):
                try:
                    # Créer l'environnement et l'algorithme
                    env = env_class(**env_params)
                    algo = algorithm_class(env, **params)
                    
                    # Entraîner
                    algo.train(verbose=False)
                    
                    # Évaluer
                    eval_results = algo.evaluate_policy(num_episodes=num_eval_episodes)
                    score = eval_results[evaluation_metric]
                    trial_scores.append(score)
                    
                except Exception as e:
                    print(f"Erreur avec les paramètres {params}: {e}")
                    trial_scores.append(-np.inf)
                    
            # Moyenne des essais
            mean_score = np.mean(trial_scores)
            std_score = np.std(trial_scores)
            
            result = {
                'params': params,
                'mean_score': mean_score,
                'std_score': std_score,
                'trial_scores': trial_scores
            }
            all_results.append(result)
            
            if verbose:
                print(f"Score moyen: {mean_score:.3f} ± {std_score:.3f}")
                
            # Mettre à jour le meilleur
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
                
        print(f"\n{'='*50}")
        print(f"Meilleurs hyperparamètres trouvés:")
        print(f"Paramètres: {best_params}")
        print(f"Score: {best_score:.3f}")
        print(f"{'='*50}")
        
        self.results = all_results
        return best_params, all_results
        
    def random_search(self,
                     algorithm_class: Type[BaseAlgorithm],
                     env_class: Type[BaseEnvironment],
                     param_distributions: Dict[str, Callable],
                     n_iter: int = 20,
                     evaluation_metric: str = 'mean_reward',
                     num_episodes: int = 1000,
                     num_eval_episodes: int = 100,
                     env_params: Dict[str, Any] = None,
                     verbose: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Recherche aléatoire des hyperparamètres.
        
        Args:
            algorithm_class: Classe de l'algorithme
            env_class: Classe de l'environnement
            param_distributions: Distributions pour échantillonner les paramètres
            n_iter: Nombre d'itérations
            evaluation_metric: Métrique pour l'évaluation
            num_episodes: Nombre d'épisodes d'entraînement
            num_eval_episodes: Nombre d'épisodes d'évaluation
            env_params: Paramètres de l'environnement
            verbose: Affichage détaillé
            
        Returns:
            best_params: Meilleurs hyperparamètres trouvés
            all_results: Tous les résultats
        """
        if env_params is None:
            env_params = {}
            
        print(f"\nRecherche aléatoire des hyperparamètres")
        print(f"Algorithme: {algorithm_class.__name__}")
        print(f"Environnement: {env_class.__name__}")
        print(f"Nombre d'itérations: {n_iter}")
        
        all_results = []
        best_score = -np.inf
        best_params = None
        
        for i in tqdm(range(n_iter), desc="Itérations"):
            # Échantillonner les paramètres
            params = {}
            for param_name, distribution in param_distributions.items():
                params[param_name] = distribution()
                
            params['episodes'] = num_episodes
            
            if verbose:
                print(f"\nItération {i+1}: {params}")
                
            try:
                # Créer l'environnement et l'algorithme
                env = env_class(**env_params)
                algo = algorithm_class(env, **params)
                
                # Entraîner
                algo.train(verbose=False)
                
                # Évaluer
                eval_results = algo.evaluate_policy(num_episodes=num_eval_episodes)
                score = eval_results[evaluation_metric]
                
                result = {
                    'params': params,
                    'score': score,
                    'eval_results': eval_results
                }
                all_results.append(result)
                
                if verbose:
                    print(f"Score: {score:.3f}")
                    
                # Mettre à jour le meilleur
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
            except Exception as e:
                print(f"Erreur: {e}")
                
        print(f"\n{'='*50}")
        print(f"Meilleurs hyperparamètres trouvés:")
        print(f"Paramètres: {best_params}")
        print(f"Score: {best_score:.3f}")
        print(f"{'='*50}")
        
        self.results = all_results
        return best_params, all_results
        
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyse les résultats de l'optimisation.
        
        Returns:
            Dictionnaire avec l'analyse
        """
        if not self.results:
            return {}
            
        # Trier par score
        sorted_results = sorted(self.results, 
                              key=lambda x: x.get('mean_score', x.get('score', -np.inf)), 
                              reverse=True)
        
        # Top 5
        top_5 = sorted_results[:5]
        
        # Analyse par paramètre
        param_analysis = {}
        
        # Extraire tous les noms de paramètres
        all_params = set()
        for result in self.results:
            all_params.update(result['params'].keys())
            
        for param_name in all_params:
            if param_name == 'episodes':
                continue
                
            values = []
            scores = []
            
            for result in self.results:
                if param_name in result['params']:
                    values.append(result['params'][param_name])
                    scores.append(result.get('mean_score', result.get('score', -np.inf)))
                    
            param_analysis[param_name] = {
                'values': values,
                'scores': scores,
                'correlation': np.corrcoef(values, scores)[0, 1] if len(set(values)) > 1 else 0
            }
            
        return {
            'top_5': top_5,
            'param_analysis': param_analysis,
            'total_experiments': len(self.results)
        } 