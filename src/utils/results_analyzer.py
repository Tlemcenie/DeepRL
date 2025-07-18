"""
Analyseur de résultats
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import scipy.stats as stats
from collections import defaultdict


class ResultsAnalyzer:
    """
    Classe pour analyser les résultats des expérimentations.
    """
    
    def __init__(self):
        self.results = None
        
    def load_results(self, results: List[Dict[str, Any]]):
        """
        Charge les résultats pour l'analyse.
        
        Args:
            results: Liste des résultats d'expérimentation
        """
        self.results = results
        
    def create_summary_dataframe(self) -> pd.DataFrame:
        """
        Crée un DataFrame récapitulatif des résultats.
        
        Returns:
            DataFrame avec le résumé
        """
        if not self.results:
            return pd.DataFrame()
            
        summary_data = []
        
        for result in self.results:
            if 'error' not in result and 'evaluation_results' in result:
                eval_res = result['evaluation_results']
                summary_data.append({
                    'Algorithm': result['algorithm'],
                    'Environment': result['environment'],
                    'Mean Reward': eval_res['mean_reward'],
                    'Std Reward': eval_res['std_reward'],
                    'Min Reward': eval_res.get('min_reward', np.nan),
                    'Max Reward': eval_res.get('max_reward', np.nan),
                    'Success Rate': eval_res.get('success_rate', np.nan),
                    'Mean Episode Length': eval_res.get('mean_episode_length', np.nan),
                    'Training Time': result.get('training_time', np.nan),
                    'Episodes': result.get('algorithm_params', {}).get('episodes', np.nan)
                })
                
        return pd.DataFrame(summary_data)
        
    def statistical_comparison(self, metric: str = 'Mean Reward') -> Dict[str, Any]:
        """
        Effectue une comparaison statistique entre algorithmes.
        
        Args:
            metric: Métrique à comparer
            
        Returns:
            Dictionnaire avec les résultats statistiques
        """
        df = self.create_summary_dataframe()
        
        if df.empty:
            return {}
            
        results = {
            'metric': metric,
            'algorithms': {},
            'pairwise_tests': {},
            'anova': None
        }
        
        # Statistiques par algorithme
        for algo in df['Algorithm'].unique():
            algo_data = df[df['Algorithm'] == algo][metric].dropna()
            results['algorithms'][algo] = {
                'mean': algo_data.mean(),
                'std': algo_data.std(),
                'median': algo_data.median(),
                'count': len(algo_data)
            }
            
        # ANOVA si plus de 2 algorithmes
        algo_groups = []
        algo_names = []
        
        for algo in df['Algorithm'].unique():
            data = df[df['Algorithm'] == algo][metric].dropna()
            if len(data) > 0:
                algo_groups.append(data)
                algo_names.append(algo)
                
        if len(algo_groups) > 2:
            f_stat, p_value = stats.f_oneway(*algo_groups)
            results['anova'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
        # Tests pairwise (t-test)
        for i in range(len(algo_names)):
            for j in range(i+1, len(algo_names)):
                algo1, algo2 = algo_names[i], algo_names[j]
                data1 = df[df['Algorithm'] == algo1][metric].dropna()
                data2 = df[df['Algorithm'] == algo2][metric].dropna()
                
                if len(data1) > 1 and len(data2) > 1:
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    
                    results['pairwise_tests'][f"{algo1} vs {algo2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'better': algo1 if data1.mean() > data2.mean() else algo2
                    }
                    
        return results
        
    def best_algorithm_per_environment(self) -> pd.DataFrame:
        """
        Trouve le meilleur algorithme pour chaque environnement.
        
        Returns:
            DataFrame avec les meilleurs algorithmes
        """
        df = self.create_summary_dataframe()
        
        if df.empty:
            return pd.DataFrame()
            
        best_algos = []
        
        for env in df['Environment'].unique():
            env_data = df[df['Environment'] == env]
            
            # Trouver le meilleur par récompense moyenne
            best_idx = env_data['Mean Reward'].idxmax()
            best_row = env_data.loc[best_idx]
            
            best_algos.append({
                'Environment': env,
                'Best Algorithm': best_row['Algorithm'],
                'Mean Reward': best_row['Mean Reward'],
                'Success Rate': best_row['Success Rate'],
                'Training Time': best_row['Training Time']
            })
            
        return pd.DataFrame(best_algos)
        
    def convergence_analysis(self) -> Dict[str, Any]:
        """
        Analyse la convergence des algorithmes.
        
        Returns:
            Dictionnaire avec l'analyse de convergence
        """
        convergence_data = {}
        
        for result in self.results:
            if 'training_history' in result and 'rewards' in result['training_history']:
                algo = result['algorithm']
                env = result['environment']
                key = f"{algo}_{env}"
                
                rewards = result['training_history']['rewards']
                
                # Calculer la convergence
                if len(rewards) > 100:
                    # Moyenne des 100 derniers vs 100 premiers
                    early_mean = np.mean(rewards[:100])
                    late_mean = np.mean(rewards[-100:])
                    improvement = late_mean - early_mean
                    
                    # Stabilité (écart-type des 100 derniers)
                    stability = np.std(rewards[-100:])
                    
                    # Vitesse de convergence (épisode où 90% de l'amélioration est atteinte)
                    cumulative_improvement = []
                    for i in range(len(rewards)):
                        window_mean = np.mean(rewards[max(0, i-50):i+1])
                        cumulative_improvement.append(window_mean - early_mean)
                        
                    target_improvement = 0.9 * improvement
                    convergence_episode = len(rewards)
                    
                    for i, imp in enumerate(cumulative_improvement):
                        if imp >= target_improvement:
                            convergence_episode = i
                            break
                            
                    convergence_data[key] = {
                        'algorithm': algo,
                        'environment': env,
                        'early_mean': early_mean,
                        'late_mean': late_mean,
                        'improvement': improvement,
                        'stability': stability,
                        'convergence_episode': convergence_episode,
                        'convergence_ratio': convergence_episode / len(rewards)
                    }
                    
        return convergence_data
        
    def hyperparameter_sensitivity(self) -> Dict[str, Any]:
        """
        Analyse la sensibilité aux hyperparamètres.
        
        Returns:
            Dictionnaire avec l'analyse de sensibilité
        """
        sensitivity = defaultdict(lambda: defaultdict(list))
        
        # Grouper par algorithme et environnement
        for result in self.results:
            if 'error' not in result:
                algo = result['algorithm']
                env = result['environment']
                params = result.get('algorithm_params', {})
                mean_reward = result['evaluation_results']['mean_reward']
                
                for param_name, param_value in params.items():
                    if param_name not in ['episodes', 'seed']:
                        sensitivity[algo][param_name].append({
                            'value': param_value,
                            'reward': mean_reward,
                            'environment': env
                        })
                        
        # Calculer les corrélations
        sensitivity_analysis = {}
        
        for algo, params_data in sensitivity.items():
            sensitivity_analysis[algo] = {}
            
            for param_name, data_points in params_data.items():
                if len(data_points) > 3:
                    values = [d['value'] for d in data_points]
                    rewards = [d['reward'] for d in data_points]
                    
                    # Corrélation si numérique
                    if all(isinstance(v, (int, float)) for v in values):
                        correlation = np.corrcoef(values, rewards)[0, 1]
                        sensitivity_analysis[algo][param_name] = {
                            'correlation': correlation,
                            'impact': 'high' if abs(correlation) > 0.5 else 'medium' if abs(correlation) > 0.3 else 'low',
                            'direction': 'positive' if correlation > 0 else 'negative'
                        }
                        
        return sensitivity_analysis
        
    def generate_latex_table(self, df: pd.DataFrame, caption: str = "", label: str = "") -> str:
        """
        Génère un tableau LaTeX à partir d'un DataFrame.
        
        Args:
            df: DataFrame à convertir
            caption: Légende du tableau
            label: Label LaTeX
            
        Returns:
            Code LaTeX du tableau
        """
        latex = df.to_latex(index=False, float_format="%.3f", escape=False)
        
        if caption:
            latex = latex.replace('\\begin{tabular}', 
                                f'\\caption{{{caption}}}\n\\label{{tab:{label}}}\n\\begin{{tabular}}')
                                
        return latex
        
    def create_report_summary(self) -> Dict[str, Any]:
        """
        Crée un résumé complet pour le rapport.
        
        Returns:
            Dictionnaire avec toutes les analyses
        """
        return {
            'summary_df': self.create_summary_dataframe(),
            'statistical_comparison': self.statistical_comparison(),
            'best_algorithms': self.best_algorithm_per_environment(),
            'convergence_analysis': self.convergence_analysis(),
            'hyperparameter_sensitivity': self.hyperparameter_sensitivity()
        } 