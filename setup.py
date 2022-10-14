import io
import os
import re

from pathlib import Path
from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


def version():
    """ Get the local package version. """
    namespace = {}
    path = Path("z/__version__.py")
    exec(path.read_text(), namespace)
    return namespace["__version__"]


setup(
    name="sci-ztools",
    version=version(),
    url="https://github.com/zhaisilong/z",
    license='MIT',
    author="Zhai Silong",
    author_email="zhaisilong@outlook.com",
    description="A Unified Tool Set for Science Researches",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(where='.', exclude=('tests',), include=('*')),
    package_data={'z': ['vocab/*']},
    entry_points={
        'console_scripts': [
            'z = z.__main__:main',
        ],
    },
    install_requires=[
        'biopython',
        'rich',
        'jupyter',
        'ipykernel',
        'pandas',
        'scikit-learn',
        'numpy',
        'tqdm',
        'six',
        'future',
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.7',  # 对python的最低版本要求
)
