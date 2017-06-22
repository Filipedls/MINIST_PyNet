# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.txt') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='PyNet',
    version='0.1',
    description='Neural Nets from scratch',
    long_description=readme,
    author='Filipe Silva',
    author_email='filipe.dls@gmail.com',
    url='https://github.com/Filipedls/MINIST_PyNet',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

