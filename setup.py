import setuptools
from os import path

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='tm_metrics',
    version='0.1',
    author="Christian Gomes",
    author_email="christianrfg@gmail.com",
    description="Topic Modeling Quality Metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/christianrfg/tm_metrics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
         "License :: MIT License",
         "Operating System :: Ubuntu 18.04",
    ],
)
