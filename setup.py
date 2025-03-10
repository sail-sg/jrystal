"""Install script for setuptools."""

import os
# from setuptools import find_namespace_packages
from setuptools import setup, find_packages

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = 'jrystal'
VERSION = '0.0.1'
CLASSIFIERS = [
  'Environment :: Console',
  'Intended Audience :: Science/Research',
  'Intended Audience :: Developers',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python',
  'Programming Language :: Python :: 3',
  'Topic :: Software Development :: Libraries :: Python Modules',
  'Topic :: Scientific/Engineering',
]
LICENSE = 'MIT License'


def _read_requirements():
  with open(os.path.join(_CURRENT_DIR, 'requirements.txt')) as f:
    requirements = f.readlines()
  return [req.strip() for req in requirements]


setup(
  name=PACKAGE_NAME,
  version=VERSION,
  packages=find_packages(),
  description=(
    'A JAX-based Differentiable DFT Framework for Materials.'
  ),
  classifiers=CLASSIFIERS,
  license=LICENSE,
  author='Tianbo Li',
  author_email='li_tianbo@live.com',
  install_requires=_read_requirements(),
  entry_points={
    'console_scripts': [
      'jrystal=main',
    ],
  },
)
