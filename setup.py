from pathlib import Path

from setuptools import setup

setup(
    name="shard",
    version="0.1.0",
    description="Semantic Hierarchical Archive with Retrieval and Distillation",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    py_modules=["shard_core"],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.0",
        "matplotlib>=3.8.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
