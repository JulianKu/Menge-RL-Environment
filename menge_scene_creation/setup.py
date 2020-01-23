#!/usr/bin/env python3

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    version='0.1.0',
    name='menge_scene_creation',
    packages=['MengeMapParser', 'MengeMapParser.ParserUtils', 'MengeMapParser.MengeUtils'],
    package_dir={'': 'src'},
    author='Julian Kunze',
    author_email='julian-kunze@gmx.de',
)

setup(install_requires=['scikit-image', 'triangle', 'opencv-python', 'PyYAML'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      **setup_args)
