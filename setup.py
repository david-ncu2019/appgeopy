from setuptools import setup, find_packages

setup(
    name='appgeopy',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'geopandas',
        'matplotlib',
        'scipy',
        'seaborn',
        'scikit-learn',
        'prophet==1.1.1',
        'holidays==0.24'
    ],
    python_requires='>=3.8',
    author='David Nguyen',
    author_email='vinhtruongkhtn@gmail.com',
    description='A package for processing and analyzing time-series data.',
    url='https://github.com/yourusername/signal_toolbox',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
