#!/usr/bin/env python3
"""
Script pour tester un seul algorithme sur tous les environnements
Usage: python test_single_algorithm.py --algorithm QLearning --episodes 1000
"""

import sys
import os
import argparse
import json
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("⚠️ Matplotlib/Seaborn/Pandas non disponibles. Graphiques désactivés.")
    PLOTTING_AVAILABLE = False
from datetime import datetime
from typing import Dict, List, Any, Optional

# Ajouter le projet au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.environments import *
from src.algorithms import *
from src.utils import ExperimentRunner, HyperparameterTuner, ResultsAnalyzer

# Configuration des algorithmes et leurs hyperparamètres par défaut
ALGORITHM_CONFIGS = {
    # Dynamic Programming
    'PolicyIteration': {
        'class': PolicyIteration,
        'default_params': {'gamma': 0.99, 'theta': 1e-6},
        'param_ranges': {'gamma': [0.9, 0.95, 0.99], 'theta': [1e-4, 1e-5, 1e-6]}
    },
    'ValueIteration': {
        'class': ValueIteration,
        'default_params': {'gamma': 0.99, 'theta': 1e-6},
        'param_ranges': {'gamma': [0.9, 0.95, 0.99], 'theta': [1e-4, 1e-5, 1e-6]}
    },
    
    # Monte Carlo
    'OnPolicyMonteCarlo': {
        'class': OnPolicyMonteCarlo,
        'default_params': {'episodes': 1000, 'epsilon': 0.1, 'gamma': 0.99},
        'param_ranges': {'epsilon': [0.05, 0.1, 0.2], 'episodes': [500, 1000, 2000]}
    },
    'OffPolicyMonteCarlo': {
        'class': OffPolicyMonteCarlo,
        'default_params': {'episodes': 1000, 'epsilon': 0.1, 'gamma': 0.99},
        'param_ranges': {'epsilon': [0.05, 0.1, 0.2], 'episodes': [500, 1000, 2000]}
    },
    
    # Temporal Difference
    'QLearning': {
        'class': QLearning,
        'default_params': {'episodes': 1000, 'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.99},
        'param_ranges': {'alpha': [0.05, 0.1, 0.2], 'epsilon': [0.05, 0.1, 0.2]}
    },
    'Sarsa': {
        'class': Sarsa,
        'default_params': {'episodes': 1000, 'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.99},
        'param_ranges': {'alpha': [0.05, 0.1, 0.2], 'epsilon': [0.05, 0.1, 0.2]}
    },
    'ExpectedSarsa': {
        'class': ExpectedSarsa,
        'default_params': {'episodes': 1000, 'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.99},
        'param_ranges': {'alpha': [0.05, 0.1, 0.2], 'epsilon': [0.05, 0.1, 0.2]}
    },
    
    # Planning
    'DynaQ': {
        'class': DynaQ,
        'default_params': {'episodes': 500, 'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.99, 'planning_steps': 5},
        'param_ranges': {'alpha': [0.05, 0.1, 0.2], 'planning_steps': [3, 5, 10]}
    },
    'DynaQPlus': {
        'class': DynaQPlus,
        'default_params': {'episodes': 500, 'alpha': 0.1, 'epsilon': 0.1, 'gamma': 0.99, 'planning_steps': 5, 'kappa': 0.001},
        'param_ranges': {'alpha': [0.05, 0.1, 0.2], 'planning_steps': [3, 5, 10], 'kappa': [0.0001, 0.001, 0.01]}
    }
}

# Environnements disponibles
ENVIRONMENTS = [
    ('LineWorld', LineWorld),
    ('GridWorld', GridWorld),
    ('TwoRoundRockPaperScissors', TwoRoundRockPaperScissors),
    ('MontyHallLevel1', MontyHallLevel1),
    ('MontyHallLevel2', MontyHallLevel2)
]


def test_algorithm_on_all_environments(algorithm_name: str, 
                                     custom_params: Optional[Dict] = None,
                                     num_runs: int = 5,
                                     tune_hyperparameters: bool = False,
                                     save_results: bool = True,
                                     create_visualizations: bool = True) -> Dict[str, Any]:
    """
    Teste un algorithme sur tous les environnements.
    
    Args:
        algorithm_name: Nom de l'algorithme à tester
        custom_params: Paramètres personnalisés (optionnel)
        num_runs: Nombre d'exécutions par environnement
        tune_hyperparameters: Si True, effectue un tuning des hyperparamètres
        save_results: Si True, sauvegarde les résultats
        create_visualizations: Si True, crée des graphiques
    
    Returns:
        Dictionnaire contenant tous les résultats
    """
    if algorithm_name not in ALGORITHM_CONFIGS:
        raise ValueError(f"Algorithme '{algorithm_name}' non reconnu. "
                        f"Algorithmes disponibles: {list(ALGORITHM_CONFIGS.keys())}")
    
    print("="*70)
    print(f"TEST DE L'ALGORITHME: {algorithm_name}")
    print("="*70)
    
    # Configuration de l'algorithme
    config = ALGORITHM_CONFIGS[algorithm_name]
    algorithm_class = config['class']
    
    # Utiliser les paramètres personnalisés ou par défaut
    if custom_params:
        params = {**config['default_params'], **custom_params}
    else:
        params = config['default_params'].copy()
    
    print(f"Paramètres utilisés: {params}")
    print(f"Nombre d'exécutions par environnement: {num_runs}")
    print()
    
    # Créer le gestionnaire d'expérimentations
    runner = ExperimentRunner(results_dir="single_algorithm_results")
    
    # Résultats agrégés
    all_results = {
        'algorithm': algorithm_name,
        'parameters': params,
        'environments': {},
        'summary': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Tester sur chaque environnement
    for env_name, env_class in ENVIRONMENTS:
        print(f"\n{'='*50}")
        print(f"ENVIRONNEMENT: {env_name}")
        print(f"{'='*50}")
        
        env_results = {
            'runs': [],
            'hyperparameter_tuning': None,
            'statistics': {}
        }
        
        # Tuning d'hyperparamètres si demandé
        best_params = params.copy()
        if tune_hyperparameters and algorithm_name in ALGORITHM_CONFIGS:
            print("\n🔧 Tuning des hyperparamètres...")
            try:
                tuner = HyperparameterTuner()
                param_ranges = config['param_ranges']
                
                # Exécuter le tuning
                tuning_results = tuner.grid_search(
                    algorithm_class=algorithm_class,
                    env_class=env_class,
                    param_grid=param_ranges,
                    num_trials=3,  # Moins d'essais pour aller plus vite
                    num_eval_episodes=50
                )
                
                best_params, all_tuning_results = tuning_results
                env_results['hyperparameter_tuning'] = {'best_params': best_params, 'all_results': all_tuning_results}
                
                print(f"✅ Meilleurs paramètres trouvés: {best_params}")
                
            except Exception as e:
                print(f"⚠️ Erreur pendant le tuning: {e}")
                print("Utilisation des paramètres par défaut.")
        
        # Exécuter plusieurs runs avec les meilleurs paramètres
        run_results = []
        run_rewards = []
        
        for run in range(num_runs):
            print(f"\n🔄 Exécution {run + 1}/{num_runs} sur {env_name}...")
            
            try:
                # Créer l'environnement
                env = env_class()
                
                # Exécuter l'expérimentation
                result = runner.run_single_experiment(
                    algorithm_class=algorithm_class,
                    env_class=env_class,
                    algorithm_params=best_params,
                    experiment_name=f"{algorithm_name}_{env_name}_run{run+1}"
                )
                
                run_results.append(result)
                run_rewards.append(result['evaluation']['mean_reward'])
                
                print(f"✅ Récompense moyenne: {result['evaluation']['mean_reward']:.3f}")
                
            except Exception as e:
                print(f"❌ Erreur lors de l'exécution {run + 1}: {e}")
                continue
        
        # Calculer les statistiques
        if run_rewards:
            env_results['runs'] = run_results
            env_results['statistics'] = {
                'mean_reward': np.mean(run_rewards),
                'std_reward': np.std(run_rewards),
                'min_reward': np.min(run_rewards),
                'max_reward': np.max(run_rewards),
                'num_successful_runs': len(run_rewards),
                'success_rate': len(run_rewards) / num_runs
            }
            
            print(f"\n📊 STATISTIQUES pour {env_name}:")
            print(f"   Récompense moyenne: {env_results['statistics']['mean_reward']:.3f} ± {env_results['statistics']['std_reward']:.3f}")
            print(f"   Plage: [{env_results['statistics']['min_reward']:.3f}, {env_results['statistics']['max_reward']:.3f}]")
            print(f"   Runs réussis: {env_results['statistics']['num_successful_runs']}/{num_runs}")
        
        all_results['environments'][env_name] = env_results
    
    # Calculer le résumé global
    env_mean_rewards = []
    for env_name, env_data in all_results['environments'].items():
        if env_data['statistics']:
            env_mean_rewards.append(env_data['statistics']['mean_reward'])
    
    if env_mean_rewards:
        all_results['summary'] = {
            'overall_mean_reward': np.mean(env_mean_rewards),
            'overall_std_reward': np.std(env_mean_rewards),
            'num_environments_tested': len(env_mean_rewards),
            'best_environment': max(all_results['environments'].items(),
                                  key=lambda x: x[1]['statistics'].get('mean_reward', -float('inf')))[0]
        }
    
    # Sauvegarder les résultats
    if save_results:
        save_results_to_file(all_results, algorithm_name)
    
    # Créer les visualisations
    if create_visualizations:
        create_algorithm_visualizations(all_results, algorithm_name)
    
    # Affichage du résumé final
    print_final_summary(all_results)
    
    return all_results


def save_results_to_file(results: Dict, algorithm_name: str):
    """Sauvegarde les résultats dans un fichier JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"single_algorithm_results/{algorithm_name}_complete_test_{timestamp}.json"
    
    os.makedirs("single_algorithm_results", exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Résultats sauvegardés dans: {filename}")


def create_algorithm_visualizations(results: Dict, algorithm_name: str):
    """Crée des visualisations pour les résultats de l'algorithme."""
    print("\n📈 Création des visualisations...")
    
    # Données pour les graphiques
    environments = []
    mean_rewards = []
    std_rewards = []
    
    for env_name, env_data in results['environments'].items():
        if env_data['statistics']:
            environments.append(env_name)
            mean_rewards.append(env_data['statistics']['mean_reward'])
            std_rewards.append(env_data['statistics']['std_reward'])
    
    if not environments:
        print("⚠️ Aucune donnée disponible pour les visualisations.")
        return
    
    # Configuration des graphiques
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Analyse de Performance - {algorithm_name}', fontsize=16, fontweight='bold')
    
    # 1. Graphique en barres des récompenses moyennes
    ax1 = axes[0, 0]
    bars = ax1.bar(environments, mean_rewards, yerr=std_rewards, capsize=5, 
                   color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_title('Récompenses Moyennes par Environnement')
    ax1.set_ylabel('Récompense Moyenne')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    for i, (bar, mean_val) in enumerate(zip(bars, mean_rewards)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_rewards[i] + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Box plot des récompenses
    ax2 = axes[0, 1]
    all_rewards_data = []
    labels = []
    
    for env_name, env_data in results['environments'].items():
        if env_data['runs']:
            rewards = [run['evaluation']['mean_reward'] for run in env_data['runs']]
            all_rewards_data.append(rewards)
            labels.append(env_name)
    
    if all_rewards_data:
        ax2.boxplot(all_rewards_data, labels=labels)
        ax2.set_title('Distribution des Récompenses par Environnement')
        ax2.set_ylabel('Récompense')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    # 3. Heatmap des performances
    ax3 = axes[1, 0]
    if len(environments) > 1:
        # Créer une matrice avec les statistiques
        metrics = ['mean_reward', 'std_reward', 'success_rate']
        heatmap_data = []
        
        for metric in metrics:
            row = []
            for env_name in environments:
                env_stats = results['environments'][env_name]['statistics']
                row.append(env_stats.get(metric, 0))
            heatmap_data.append(row)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', 
                   xticklabels=environments, yticklabels=metrics,
                   cmap='RdYlBu_r', ax=ax3)
        ax3.set_title('Heatmap des Métriques de Performance')
    else:
        ax3.text(0.5, 0.5, 'Heatmap nécessite\nplusieurs environnements', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Heatmap des Métriques de Performance')
    
    # 4. Radar chart des performances (si plus d'un environnement)
    ax4 = axes[1, 1]
    if len(environments) > 2:
        # Normaliser les données pour le radar chart
        normalized_rewards = np.array(mean_rewards)
        normalized_rewards = (normalized_rewards - normalized_rewards.min()) / (normalized_rewards.max() - normalized_rewards.min() + 1e-8)
        
        # Angles pour le radar
        angles = np.linspace(0, 2*np.pi, len(environments), endpoint=False)
        normalized_rewards = np.concatenate([normalized_rewards, [normalized_rewards[0]]])  # Fermer le cercle
        angles = np.concatenate([angles, [angles[0]]])
        
        ax4.plot(angles, normalized_rewards, 'o-', linewidth=2, color='red')
        ax4.fill(angles, normalized_rewards, alpha=0.25, color='red')
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(environments)
        ax4.set_ylim(0, 1)
        ax4.set_title('Radar Chart - Performance Normalisée')
        ax4.grid(True)
    else:
        ax4.text(0.5, 0.5, 'Radar chart nécessite\nplus de 2 environnements', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Radar Chart - Performance Normalisée')
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"single_algorithm_results/{algorithm_name}_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Graphiques sauvegardés dans: {filename}")


def print_final_summary(results: Dict):
    """Affiche un résumé final des résultats."""
    print("\n" + "="*70)
    print("🎯 RÉSUMÉ FINAL")
    print("="*70)
    
    algorithm_name = results['algorithm']
    summary = results.get('summary', {})
    
    print(f"Algorithme testé: {algorithm_name}")
    print(f"Paramètres: {results['parameters']}")
    
    if summary:
        print(f"\nPerformance globale:")
        print(f"  Récompense moyenne: {summary['overall_mean_reward']:.3f} ± {summary['overall_std_reward']:.3f}")
        print(f"  Nombre d'environnements testés: {summary['num_environments_tested']}")
        print(f"  Meilleur environnement: {summary['best_environment']}")
    
    print(f"\nDétail par environnement:")
    for env_name, env_data in results['environments'].items():
        stats = env_data['statistics']
        if stats:
            print(f"  {env_name:<25}: {stats['mean_reward']:>7.3f} ± {stats['std_reward']:.3f} "
                  f"(réussite: {stats['success_rate']*100:.0f}%)")
        else:
            print(f"  {env_name:<25}: ❌ Échec")
    
    print("\n" + "="*70)


def main():
    """Fonction principale avec interface en ligne de commande."""
    parser = argparse.ArgumentParser(description='Teste un algorithme sur tous les environnements')
    parser.add_argument('--algorithm', '-a', required=True, 
                       choices=list(ALGORITHM_CONFIGS.keys()),
                       help='Nom de l\'algorithme à tester')
    parser.add_argument('--episodes', '-e', type=int, default=None,
                       help='Nombre d\'épisodes d\'entraînement')
    parser.add_argument('--runs', '-r', type=int, default=5,
                       help='Nombre d\'exécutions par environnement')
    parser.add_argument('--tune', action='store_true',
                       help='Effectuer un tuning des hyperparamètres')
    parser.add_argument('--no-save', action='store_true',
                       help='Ne pas sauvegarder les résultats')
    parser.add_argument('--no-viz', action='store_true',
                       help='Ne pas créer de visualisations')
    parser.add_argument('--alpha', type=float, default=None,
                       help='Learning rate (pour les algorithmes TD)')
    parser.add_argument('--epsilon', type=float, default=None,
                       help='Epsilon pour epsilon-greedy')
    parser.add_argument('--gamma', type=float, default=None,
                       help='Facteur de discount')
    
    args = parser.parse_args()
    
    # Construire les paramètres personnalisés
    custom_params = {}
    if args.episodes is not None:
        custom_params['episodes'] = args.episodes
    if args.alpha is not None:
        custom_params['alpha'] = args.alpha
    if args.epsilon is not None:
        custom_params['epsilon'] = args.epsilon
    if args.gamma is not None:
        custom_params['gamma'] = args.gamma
    
    print("🚀 Démarrage du test d'algorithme...")
    print(f"Algorithme: {args.algorithm}")
    if custom_params:
        print(f"Paramètres personnalisés: {custom_params}")
    
    try:
        results = test_algorithm_on_all_environments(
            algorithm_name=args.algorithm,
            custom_params=custom_params if custom_params else None,
            num_runs=args.runs,
            tune_hyperparameters=args.tune,
            save_results=not args.no_save,
            create_visualizations=not args.no_viz
        )
        
        print("\n✅ Test terminé avec succès!")
        
    except Exception as e:
        print(f"\n❌ Erreur pendant le test: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import numpy as np
    exit(main()) 