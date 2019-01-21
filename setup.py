import os
import setuptools
from setuptools import setup, find_packages
from distutils.core import setup

# Utility function to read the README file.

def read(fname):
	return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name = "idNet",
	version = "1.0",
	author = "Jonathan Dunn",
	author_email = "jonathan.dunn@canterbury.ac.nz",
	description = ("Language Identification, Dialect Identification"),
	license = "LGPL 3.0",
	url = "https://gitlab.com",
	keywords = "language id, dialect id",
	packages = find_packages(exclude=["*.pyc", "__pycache__"]),
	package_data={'': []},
	install_requires=["cytoolz",
						"gensim",
						"numpy",
						"pandas",
						"scipy",
						"sklearn",
						"keras"
						"numba"
						],
	include_package_data=True,
	long_description=read('README.md'),
	)