from setuptools import setup, find_packages

setup(
    name="rl_project",
    version="1.0.0",
    author="Groupe IABD",
    description="Projet de Deep Reinforcement Learning - 4A IABD",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.21.0',
        'matplotlib>=3.4.0',
        'tqdm>=4.62.0',
        'pandas>=1.3.0',
        'jupyter>=1.0.0',
        'ipywidgets>=7.6.0',
        'pygame>=2.0.0',
        'seaborn>=0.11.0',
        'scipy>=1.7.0',
        'dill>=0.3.4',
        'python-pptx>=0.6.21',
        'openpyxl>=3.0.9'
    ],
    python_requires='>=3.8',
) 