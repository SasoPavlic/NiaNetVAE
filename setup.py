from setuptools import setup

setup(
    name='NiaNetVAE',
    version="1.0.0",
    packages=['nianetvae', 'nianetvae.models', 'nianetvae.storage', 'nianetvae.dataloaders',
              'nianetvae.experiments', 'nianetvae.niapy_extension'],
    url='https://github.com/SasoPavlic/NiaNetVAE',
    license='MIT License',
    author='Saso Pavlic',
    author_email='saso.pavlic@student.um.si',
    description='Designing and constructing variational recurrent autoencoders using nature-inspired algorithms'
)
