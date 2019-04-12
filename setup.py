import setuptools

# get readme
with open('README.md', 'r') as fh:
    long_description = fh.read()

# get version
version = {}
with open("simf/version.py") as fp:
    exec(fp.read(), version)
VERSION = version['__version__']

setuptools.setup(
    name='simf',
    version=VERSION,
    author='Martin Jakomin',
    author_email='martin.jakomin@fri.uni-lj.si',
    description='SIMF - Simultaneous Incremental Matrix Factorization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MartinJakomin/SIMF',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'tables',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

