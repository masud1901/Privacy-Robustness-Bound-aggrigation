#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="fedlearn",
    version="0.1.0",
    author="Privacy-Preserving FL Team",
    author_email="team@example.com",
    description="A package for privacy-robust federated learning with ResNet-18",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/privacy_preserving_FL",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
) 