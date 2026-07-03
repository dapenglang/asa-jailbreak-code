from setuptools import setup, find_packages

setup(
    name="asa-jailbreak",
    version="1.0.0",
    description="Active Subspace Attack: Exploiting Spectral Geometry of Loss Landscapes for Efficient LLM Jailbreaking",
    author="Yan Wang, Dapeng Lang",
    author_email="wangyan0566@hit.edu.cn",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "datasets>=2.14.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0.0",
    ],
)
