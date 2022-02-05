#!/usr/bin/env python
import os
from setuptools import setup, find_packages

# src_dir = os.path.join(os.getcwd(), 'hdcob')
# packages = {"" : "src"}
# for package in find_packages("hdcob"):
#     packages[package] = "hdcob"

setup(
    name='hdcob',
    version='0.0.21',
    url='https://gitlab.inria.fr/epione/hdcob',
    author='Jaume Banus',
    author_email='jaume.banus-cobo@inria.fr',
    description='Heart dynamics conditioned on brain',
    packages=["hdcob",
              "hdcob/virca",
              "hdcob/gp",
              "hdcob/vi",
              "hdcob/utilities"],
)
