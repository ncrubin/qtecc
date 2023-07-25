"""
QTECC installation script
"""

import io
import re

from setuptools import setup, find_packages


def version_number(path: str) -> str:
    """Get qcpanop's version number from the src directory
    """
    exp = r'__version__[ ]*=[ ]*["|\']([\d]+\.[\d]+\.[\d]+[\.dev[\d]*]?)["|\']'
    version_re = re.compile(exp)

    with open(path, 'r') as fqe_version:
        version = version_re.search(fqe_version.read()).group(1)

    return version


def main() -> None:
    """
    """
    version_path = 'qtecc/_version.py'

    __version__ = version_number(version_path)

    if __version__ is None:
        raise ValueError('Version information not found in ' + version_path)

    long_description = ('=====\n' +
                        'QTECC\n' +
                        '=====\n')
    stream = io.open('README.md', encoding='utf-8')
    stream.readline()
    long_description += stream.read()

    requirements_buffer = open('requirements.txt').readlines()
    requirements = [r.strip() for r in requirements_buffer]

    setup(
        name='qtecc',
        version=__version__,
        author='Nicholas C. Rubin',
        author_email='rubinnc0@gmail.com',
        description='Quantum Tailored-Extended Coupled Cluster',
        long_description=long_description,
        install_requires=None,
        license='Apache 2',
        packages=find_packages(),
        )


main()