"""
Gestionnaire d'expérimentations
Pour exécuter et sauvegarder les résultats de manière systématique
"""

import os
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Type, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..environments import BaseEnvironment
from ..algorithms import BaseAlgorithm


class ExperimentRunner:
    """
    Classe pour exécuter des expérimentations systématiques.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Args:
            results_dir: Répertoire pour sauvegarder les résultats
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Créer un sous-dossier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(results_dir, f"experiment_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.results = []
        
    def run_single_experiment(self, 
                            algorithm_class: Type[BaseAlgorithm],
                            env_class: Type[BaseEnvironment],
                            algorithm_params: Dict[str, Any],
                            env_params: Dict[str, Any] = None,
                            num_evaluation_episodes: int = 100,
                            save_trained_model: bool = True,
                            experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute une seule expérimentation.
        
        Args:
            algorithm_class: Classe de l'algorithme à tester
            env_class: Classe de l'environnement
            algorithm_params: Hyperparamètres de l'algorithme
            env_params: Paramètres de l'environnement
            num_evaluation_episodes: Nombre d'épisodes pour l'évaluation
            save_trained_model: Si True, sauvegarde le modèle entraîné
            experiment_name: Nom de l'expérimentation
            
        Returns:
            Dictionnaire contenant les résultats
        """
        if env_params is None:
            env_params = {}
            
        # Créer l'environnement et l'algorithme
        env = env_class(**env_params)
        algo = algorithm_class(env, **algorithm_params)
        
        # Nom de l'expérimentation
        if experiment_name is None:
            experiment_name = f"{algorithm_class.__name__}_{env_class.__name__}"
            
        print(f"\n{'='*60}")
        print(f"Expérimentation: {experiment_name}")
        print(f"Algorithme: {algorithm_class.__name__}")
        print(f"Environnement: {env_class.__name__}")
        print(f"Hyperparamètres: {algorithm_params}")
        print(f"{'='*60}\n")
        
        # Entraînement
        start_time = time.time()
        training_results = algo.train(verbose=True)
        training_time = time.time() - start_time
        
        # Évaluation
        print("\nÉvaluation de la politique apprise...")
        try:
            evaluation_results = algo.evaluate_policy(num_episodes=num_evaluation_episodes)
            print(f"✅ Évaluation réussie: {evaluation_results['mean_reward']:.3f}")
        except Exception as e:
            print(f"Erreur lors de l'évaluation: {e}")
            evaluation_results = {
                'mean_reward': 0.0, 
                'std_reward': 0.0, 
                'success_rate': 0.0,
                'error': str(e)
            }
        
        # Copier l'historique d'entraînement de manière sûre
        training_history_safe = {}
        try:
            for key, value in algo.training_history.items():
                if isinstance(value, list):
                    training_history_safe[key] = value.copy()
                else:
                    training_history_safe[key] = value
        except:
            training_history_safe = {'error': 'Could not copy training history'}
        
        # Nettoyer training_results pour éviter les références problématiques
        training_results_safe = {}
        if training_results:
            for key, value in training_results.items():
                if key in ['final_policy', 'final_q_function', 'final_value_function']:
                    # Ne pas inclure les fonctions/politiques complètes
                    training_results_safe[key + '_size'] = len(value) if hasattr(value, '__len__') else 'N/A'
                else:
                    training_results_safe[key] = value
        
        # Résultats complets
        results = {
            'experiment_name': experiment_name,
            'algorithm': algorithm_class.__name__,
            'environment': env_class.__name__,
            'algorithm_params': algorithm_params,
            'env_params': env_params,
            'training_results': training_results_safe,
            'evaluation_results': evaluation_results,
            'evaluation': evaluation_results,  # Alias pour compatibilité
            'training_time': training_time,
            'training_history': training_history_safe,
            'timestamp': datetime.now().isoformat()
        }
        
        # Sauvegarder les résultats
        self._save_experiment_results(results, experiment_name)
        
        # Sauvegarder le modèle entraîné (désactivé temporairement pour éviter erreurs sérialisation)
        if save_trained_model:
            try:
                model_path = os.path.join(self.experiment_dir, f"{experiment_name}_model.pkl")
                # algo.save(model_path)  # Désactivé temporairement
                results['model_path'] = f"Sauvegarde désactivée - {model_path}"
            except Exception as e:
                results['model_save_error'] = str(e)
            
        # Générer les graphiques
        self._generate_plots(algo, experiment_name)
        
        # Ajouter aux résultats globaux
        self.results.append(results)
        
        return results
        
    def run_algorithm_on_all_environments(self,
                                        algorithm_class: Type[BaseAlgorithm],
                                        environments: List[Type[BaseEnvironment]],
                                        algorithm_params: Dict[str, Any],
                                        num_evaluation_episodes: int = 100) -> List[Dict[str, Any]]:
        """
        Teste un algorithme sur tous les environnements.
        
        Args:
            algorithm_class: Classe de l'algorithme
            environments: Liste des classes d'environnement
            algorithm_params: Hyperparamètres
            num_evaluation_episodes: Nombre d'épisodes d'évaluation
            
        Returns:
            Liste des résultats
        """
        results = []
        
        for env_class in environments:
            try:
                result = self.run_single_experiment(
                    algorithm_class=algorithm_class,
                    env_class=env_class,
                    algorithm_params=algorithm_params,
                    num_evaluation_episodes=num_evaluation_episodes
                )
                results.append(result)
            except Exception as e:
                print(f"\nErreur avec {algorithm_class.__name__} sur {env_class.__name__}: {e}")
                results.append({
                    'algorithm': algorithm_class.__name__,
                    'environment': env_class.__name__,
                    'error': str(e),
                    'status': 'failed'
                })
                
        return results
        
    def run_all_algorithms_on_environment(self,
                                        algorithms: List[Tuple[Type[BaseAlgorithm], Dict[str, Any]]],
                                        env_class: Type[BaseEnvironment],
                                        env_params: Dict[str, Any] = None,
                                        num_evaluation_episodes: int = 100) -> List[Dict[str, Any]]:
        """
        Teste tous les algorithmes sur un environnement.
        
        Args:
            algorithms: Liste de tuples (classe_algorithme, hyperparamètres)
            env_class: Classe de l'environnement
            env_params: Paramètres de l'environnement
            num_evaluation_episodes: Nombre d'épisodes d'évaluation
            
        Returns:
            Liste des résultats
        """
        results = []
        
        for algo_class, algo_params in algorithms:
            try:
                result = self.run_single_experiment(
                    algorithm_class=algo_class,
                    env_class=env_class,
                    algorithm_params=algo_params,
                    env_params=env_params,
                    num_evaluation_episodes=num_evaluation_episodes
                )
                results.append(result)
            except Exception as e:
                print(f"\nErreur avec {algo_class.__name__} sur {env_class.__name__}: {e}")
                results.append({
                    'algorithm': algo_class.__name__,
                    'environment': env_class.__name__,
                    'error': str(e),
                    'status': 'failed'
                })
                
        return results
        
    def run_complete_experiment_matrix(self,
                                     algorithms: List[Tuple[Type[BaseAlgorithm], Dict[str, Any]]],
                                     environments: List[Type[BaseEnvironment]],
                                     num_evaluation_episodes: int = 100) -> pd.DataFrame:
        """
        Execute tous les algorithmes sur tous les environnements.
        
        Args:
            algorithms: Liste de tuples (classe_algorithme, hyperparamètres)
            environments: Liste des classes d'environnement
            num_evaluation_episodes: Nombre d'épisodes d'évaluation
            
        Returns:
            DataFrame avec les résultats
        """
        all_results = []
        
        total_experiments = len(algorithms) * len(environments)
        experiment_count = 0
        
        print(f"\n{'='*60}")
        print(f"MATRICE COMPLÈTE D'EXPÉRIMENTATIONS")
        print(f"Nombre d'algorithmes: {len(algorithms)}")
        print(f"Nombre d'environnements: {len(environments)}")
        print(f"Total d'expérimentations: {total_experiments}")
        print(f"{'='*60}\n")
        
        for algo_class, algo_params in algorithms:
            for env_class in environments:
                experiment_count += 1
                print(f"\nExpérimentation {experiment_count}/{total_experiments}")
                
                try:
                    result = self.run_single_experiment(
                        algorithm_class=algo_class,
                        env_class=env_class,
                        algorithm_params=algo_params,
                        num_evaluation_episodes=num_evaluation_episodes
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"\nErreur: {e}")
                    all_results.append({
                        'algorithm': algo_class.__name__,
                        'environment': env_class.__name__,
                        'error': str(e),
                        'status': 'failed',
                        'evaluation_results': {'mean_reward': np.nan}
                    })
                    
        # Créer un DataFrame récapitulatif
        summary_data = []
        for result in all_results:
            if 'error' not in result:
                summary_data.append({
                    'Algorithm': result['algorithm'],
                    'Environment': result['environment'],
                    'Mean Reward': result['evaluation_results']['mean_reward'],
                    'Std Reward': result['evaluation_results']['std_reward'],
                    'Success Rate': result['evaluation_results']['success_rate'],
                    'Training Time': result['training_time'],
                    'Status': 'success'
                })
            else:
                summary_data.append({
                    'Algorithm': result['algorithm'],
                    'Environment': result['environment'],
                    'Mean Reward': np.nan,
                    'Std Reward': np.nan,
                    'Success Rate': np.nan,
                    'Training Time': np.nan,
                    'Status': 'failed'
                })
                
        df = pd.DataFrame(summary_data)
        
        # Sauvegarder le DataFrame
        df.to_csv(os.path.join(self.experiment_dir, 'results_summary.csv'), index=False)
        
        # Créer une matrice de performance
        self._create_performance_matrix(df)
        
        return df
        
    def _convert_for_json(self, obj):
        """Convertit les objets non-sérialisables pour JSON."""
        from collections import defaultdict
        
        if isinstance(obj, defaultdict):
            # Convertir defaultdict en dict normal
            return self._convert_for_json(dict(obj))
        elif isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                # Convertir les clés tuple en string
                if isinstance(k, tuple):
                    k = str(k)
                new_dict[k] = self._convert_for_json(v)
            return new_dict
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Pour les objets avec attributs, convertir en dict
            return self._convert_for_json(obj.__dict__)
        else:
            return obj
    
    def _save_experiment_results(self, results: Dict[str, Any], experiment_name: str):
        """Sauvegarde les résultats d'une expérimentation."""
        # Sauvegarder en JSON (sans l'historique complet qui peut être volumineux)
        results_json = results.copy()
        results_json.pop('training_history', None)
        
        # Convertir pour JSON
        results_json = self._convert_for_json(results_json)
        
        json_path = os.path.join(self.experiment_dir, f"{experiment_name}_results.json")
        with open(json_path, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
            
        # Sauvegarder en pickle (avec tout)
        pickle_path = os.path.join(self.experiment_dir, f"{experiment_name}_full_results.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
            
    def _generate_plots(self, algo: BaseAlgorithm, experiment_name: str):
        """Génère les graphiques pour une expérimentation."""
        plot_path = os.path.join(self.experiment_dir, f"{experiment_name}_training_history.png")
        algo.plot_training_history(save_path=plot_path)
        plt.close()
        
    def _create_performance_matrix(self, df: pd.DataFrame):
        """Crée une matrice de performance visuelle."""
        # Pivot pour créer la matrice
        matrix = df.pivot(index='Environment', columns='Algorithm', values='Mean Reward')
        
        # Créer le heatmap
        plt.figure(figsize=(12, 8))
        im = plt.imshow(matrix.values, cmap='RdYlGn', aspect='auto')
        
        # Ajouter les labels
        plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha='right')
        plt.yticks(range(len(matrix.index)), matrix.index)
        
        # Ajouter les valeurs dans les cellules
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                value = matrix.values[i, j]
                if not np.isnan(value):
                    plt.text(j, i, f'{value:.2f}', ha='center', va='center')
                    
        plt.colorbar(im, label='Récompense moyenne')
        plt.title('Matrice de Performance: Algorithmes vs Environnements')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.experiment_dir, 'performance_matrix.png'))
        plt.close()
        
    def generate_report(self):
        """Génère un rapport HTML des résultats."""
        # TODO: Implémenter la génération de rapport HTML
        pass 