from distutils.core import setup

setup(
    name='mednickdb_pysleep',
    version='0.2dev',
    packages=['mednickdb_pysleep'],
    license='MIT',
    long_description=open('README.md').read(),
    requires=['numpy']
)