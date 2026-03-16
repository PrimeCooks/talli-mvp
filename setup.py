"""
TALLI — Task-Adaptive LLM Inference
Setup configuration
"""
from setuptools import setup, find_packages

setup(
    name="talli-llm",
    version="0.1.0",
    description="Task-Adaptive LLM Inference — Run large models on low VRAM",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="AI Agent Office Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
    ],
    extras_require={
        "embeddings": ["sentence-transformers>=2.2.0"],
    },
    entry_points={
        "console_scripts": [
            "talli=talli.cli:main",
            "talli-server=talli.server:run_server",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
)
