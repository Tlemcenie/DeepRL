"""
Script principal pour lancer les expérimentations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments import *
from src.algorithms import *
from src.utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    """Fonction principale pour lancer toutes les expérimentations."""
    
    print("="*60)
    print("PROJET DEEP REINFORCEMENT LEARNING - 4A IABD")
    print("="*60)
    
    # Créer le gestionnaire d'expérimentations
    runner = ExperimentRunner(results_dir="results")
    
    # Définir les environnements à tester
    environments = [
        LineWorld,
        GridWorld,
        TwoRoundRockPaperScissors,
        MontyHallLevel1,
        MontyHallLevel2
    ]
    
    # Définir les algorithmes et leurs hyperparamètres
    algorithms = [
        # Dynamic Programming
        (PolicyIteration, {
            'gamma': 0.99,
            'theta': 1e-6,
            'episodes': 1  # Pas utilisé pour DP
        }),
        (ValueIteration, {
            'gamma': 0.99,
            'theta': 1e-6,
            'episodes': 1  # Pas utilisé pour DP
        }),
        
        # Monte Carlo
        (MonteCarloES, {
            'gamma': 0.99,
            'episodes': 5000,
            'alpha': None  # Moyenne
        }),
        (OnPolicyMonteCarlo, {
            'gamma': 0.99,
            'episodes': 5000,
            'epsilon': 0.1,
            'alpha': None
        }),
        (OffPolicyMonteCarlo, {
            'gamma': 0.99,
            'episodes': 5000,
            'behavior_epsilon': 0.3
        }),
        
        # Temporal Difference
        (Sarsa, {
            'gamma': 0.99,
            'episodes': 2000,
            'alpha': 0.1,
            'epsilon': 0.1,
            'epsilon_decay': 0.995
        }),
        (QLearning, {
            'gamma': 0.99,
            'episodes': 2000,
            'alpha': 0.1,
            'epsilon': 0.1,
            'epsilon_decay': 0.995
        }),
        (ExpectedSarsa, {
            'gamma': 0.99,
            'episodes': 2000,
            'alpha': 0.1,
            'epsilon': 0.1,
            'epsilon_decay': 0.995
        }),
        
        # Planning
        (DynaQ, {
            'gamma': 0.99,
            'episodes': 500,
            'alpha': 0.1,
            'epsilon': 0.1,
            'planning_steps': 5
        }),
        (DynaQPlus, {
            'gamma': 0.99,
            'episodes': 500,
            'alpha': 0.1,
            'epsilon': 0.1,
            'planning_steps': 5,
            'kappa': 0.001
        })
    ]
    
    # Option 1: Tester tous les algorithmes sur tous les environnements
    print("\n1. Lancement de la matrice complète d'expérimentations...")
    results_df = runner.run_complete_experiment_matrix(
        algorithms=algorithms,
        environments=environments,
        num_evaluation_episodes=100
    )
    
    print("\n2. Résumé des résultats:")
    print(results_df)
    
    # Analyser les résultats
    analyzer = ResultsAnalyzer()
    analyzer.load_results(runner.results)
    
    # Créer le rapport d'analyse
    print("\n3. Analyse des résultats...")
    report = analyzer.create_report_summary()
    
    # Afficher les meilleurs algorithmes par environnement
    print("\n4. Meilleurs algorithmes par environnement:")
    best_algos = analyzer.best_algorithm_per_environment()
    print(best_algos)
    
    # Visualisations
    visualizer = Visualizer()
    
    # Courbes d'apprentissage
    print("\n5. Génération des visualisations...")
    visualizer.plot_learning_curves(
        runner.results,
        title="Courbes d'apprentissage - Tous les algorithmes",
        save_path=os.path.join(runner.experiment_dir, "learning_curves.png")
    )
    
    # Comparaison des performances
    visualizer.plot_performance_comparison(
        results_df,
        metric='Mean Reward',
        save_path=os.path.join(runner.experiment_dir, "performance_comparison.png")
    )
    
    print(f"\n6. Résultats sauvegardés dans: {runner.experiment_dir}")
    
    # Option 2: Optimisation des hyperparamètres sur un environnement spécifique
    print("\n7. Optimisation des hyperparamètres pour Q-Learning sur LineWorld...")
    tuner = HyperparameterTuner()
    
    param_grid = {
        'alpha': [0.01, 0.1, 0.5],
        'epsilon': [0.01, 0.1, 0.3],
        'gamma': [0.9, 0.95, 0.99]
    }
    
    best_params, tuning_results = tuner.grid_search(
        algorithm_class=QLearning,
        env_class=LineWorld,
        param_grid=param_grid,
        num_episodes=1000,
        num_trials=3,
        verbose=False
    )
    
    print(f"Meilleurs hyperparamètres trouvés: {best_params}")
    
    print("\n✅ Expérimentations terminées!")
    
    return runner, analyzer, visualizer


def test_secret_environments():
    """Tester les environnements secrets si disponibles."""
    
    print("\n" + "="*60)
    print("TEST DES ENVIRONNEMENTS SECRETS")
    print("="*60)
    
    try:
        from src.environments.secret_envs_integration import get_secret_environments
        
        secret_envs = get_secret_environments()
        
        if not secret_envs:
            print("⚠️ Aucun environnement secret disponible.")
            return
            
        print(f"✅ {len(secret_envs)} environnements secrets trouvés!")
        
        # Tester chaque environnement secret avec les meilleurs algorithmes
        runner = ExperimentRunner(results_dir="results_secrets")
        
        # Algorithmes à tester sur les secrets
        algorithms_for_secrets = [
            (QLearning, {
                'gamma': 0.99,
                'episodes': 10000,
                'alpha': 0.1,
                'epsilon': 0.2,
                'epsilon_decay': 0.995
            }),
            (DynaQ, {
                'gamma': 0.99,
                'episodes': 5000,
                'alpha': 0.1,
                'epsilon': 0.1,
                'planning_steps': 10
            })
        ]
        
        for env_name, env_class in secret_envs.items():
            print(f"\nTest de {env_name}...")
            
            for algo_class, params in algorithms_for_secrets:
                try:
                    result = runner.run_single_experiment(
                        algorithm_class=algo_class,
                        env_class=env_class,
                        algorithm_params=params,
                        num_evaluation_episodes=100,
                        experiment_name=f"{algo_class.__name__}_{env_name}"
                    )
                    
                    print(f"✅ {algo_class.__name__} sur {env_name}: "
                          f"Récompense moyenne = {result['evaluation_results']['mean_reward']:.2f}")
                          
                except Exception as e:
                    print(f"❌ Erreur avec {algo_class.__name__} sur {env_name}: {e}")
                    
    except ImportError as e:
        print(f"⚠️ Impossible d'importer les environnements secrets: {e}")
        

def demonstrate_policies():
    """Démontrer les politiques apprises sur quelques environnements."""
    
    print("\n" + "="*60)
    print("DÉMONSTRATION DES POLITIQUES APPRISES")
    print("="*60)
    
    # 1. Q-Learning sur LineWorld
    print("\n1. Q-Learning sur LineWorld:")
    env = LineWorld(size=7)
    algo = QLearning(env, gamma=0.99, episodes=1000, alpha=0.1, epsilon=0.1)
    algo.train(verbose=False)
    algo.demonstrate_policy(num_steps=10, delay=0)
    
    # 2. Policy Iteration sur GridWorld
    print("\n2. Policy Iteration sur GridWorld:")
    env = GridWorld(width=4, height=4)
    algo = PolicyIteration(env, gamma=0.99)
    algo.train(verbose=False)
    
    # Afficher la politique
    print("\nPolitique optimale trouvée:")
    policy = algo.get_policy()
    for y in range(4):
        row = ""
        for x in range(4):
            if (x, y) in policy:
                action = policy[(x, y)]
                symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
                row += symbols.get(action, '?') + " "
            else:
                row += ". "
        print(row)
        

if __name__ == "__main__":
    # Lancer les expérimentations principales
    runner, analyzer, visualizer = main()
    
    # Tester les environnements secrets
    test_secret_environments()
    
    # Démontrer quelques politiques
    demonstrate_policies()
    
    print("\n🎉 Projet terminé avec succès!") 