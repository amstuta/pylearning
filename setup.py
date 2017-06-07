from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as fd:
    long_description = fd.read()

setup(
    name='pylearning',

    version='2.1.0b1',

    description='Simple high-level library to use decision trees and random forest learners',
    long_description=long_description,

    url="https://github.com/amstuta/pylearning.git",

    author="Arthur Amstutz",
    author_email="arthur.amstutz@gmail.com",

    license='MIT',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],

    keywords='machine learning data decision trees random forest',

    packages=find_packages(exclude=['contrib','docs','tests'])
)
