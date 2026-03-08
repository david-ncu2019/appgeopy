"""
Backward-compatible setup.py. The canonical package metadata lives in pyproject.toml.
This file is kept so that older tools (e.g., pip < 21, some CI systems) can still
install the package without pyproject.toml support.
"""
from setuptools import setup, find_packages

setup(
    name='appgeopy',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'seaborn',
        'scikit-learn',
        'xarray',
        'ruptures',
        'h5py',
        'openpyxl',
        'pyproj',
        'shapely',
    ],
    extras_require={
        'geo': ['geopandas', 'fiona', 'GDAL'],
        'timeseries': ['prophet==1.1.1', 'holidays==0.24'],
        'full': ['geopandas', 'fiona', 'GDAL', 'prophet==1.1.1', 'holidays==0.24'],
        'dev': ['pytest', 'pytest-cov', 'black', 'ruff', 'mypy'],
    },
    python_requires='>=3.8',
    author='David Nguyen',
    author_email='vinhtruongkhtn@gmail.com',
    description='A Python package for processing and analyzing geospatial time-series data.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/david-ncu2019/appgeopy',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: GIS',
    ],
)
