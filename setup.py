from setuptools import setup, find_packages

setup(
    name="protoInterpretation",
    version="0.1.0",
    description="A lightweight library for sampling LLM outputs to analyze how the horizon of possible output changes with different prompts",
    author="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "umap-learn>=0.5.3",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
)

