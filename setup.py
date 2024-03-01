from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'


def get_requirements(filepath: str) -> List[str]:
    '''
    Return list of requirements from file.
    '''
    requirements = []
    with open(filepath, 'r') as f:
        requirements = f.readlines()
        # replace '\n' with '', readlines() add '\n'
        requirements = [req.replace('\n', '') for req in requirements]

        # -e . should not be included
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='ml-project',
    version='0.0.1',
    author='Vedant Mahida',
    author_email='vedantsinh13@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)
