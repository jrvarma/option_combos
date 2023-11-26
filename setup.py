#!/usr/bin/env python
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ['numpy', 'pandas', 'scipy']

setup(name='Black_Scholes',
      version='0.9.1',
      author='Jayanth R. Varma, Vineet Virmani',
      maintainer='Jayanth R. Varma',
      maintainer_email='jrvarma@gmail.com',
      description='Black Scholes for options, portfolios, combos',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/jrvarma/Black_Scholes",
      packages=['Black_Scholes'],
      install_requires=install_requires,
      extras_require={
      },
      entry_points={
      },
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Environment :: Console",
          "Intended Audience :: End Users/Desktop",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Programming Language :: Python :: 3",
      ],
      python_requires='>=3.0',
)  # noqa E124
