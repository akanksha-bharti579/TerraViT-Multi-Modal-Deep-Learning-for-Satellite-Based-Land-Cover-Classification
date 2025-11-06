"""
TerraViT: Multi-modal Deep Learning for Earth Observation
Setup configuration for package installation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="terravit",
    version="1.0.0",
    author="Akanksha Bharti",
    author_email="akankshabharti12379@gmail.com",
    description="Advanced multi-modal deep learning framework for satellite imagery analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/terravit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "jupyter>=1.0",
        ],
    },
    # Note: Console scripts commented out as training scripts are in examples/
    # entry_points={
    #     "console_scripts": [
    #         "terravit-train=src.training.train:main",
    #         "terravit-eval=src.training.evaluate:main",
    #     ],
    # },
)

