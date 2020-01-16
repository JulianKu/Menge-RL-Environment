#!/usr/bin/env python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    name='menge_scene_creation',
    packages=['MengeMapParser'],
    package_dir={'': 'src'},
    author='Julian Kunze',
    author_email='julian-kunze@gmx.de',
)

setup(requires=['scikit-image', 'triangle', 'cv2', 'PyYAML'], **setup_args)
