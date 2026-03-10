from setuptools import setup, find_packages

setup(
    name="minigenie",
    version="0.1.0",
    description="Action-conditioned video world model using flow matching, built from scratch in PyTorch.",
    url="https://huggingface.co/spaces/BrutalCaesar/minigenie",
    author="minigenie",
    packages=find_packages(),
    python_requires=">=3.10",
)
