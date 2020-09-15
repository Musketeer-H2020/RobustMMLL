from setuptools import setup, find_packages

setup(
    name='RobustMMLL',
    version='0.1.0',
    description='Package for interacting with RobustMMLL',
    author='marferdiaz',
    author_email='marcos.fernandez@treelogic.com',
    license='GPLv3',
    packages=find_packages('.'),
    python_requires='>=3.6',
    install_requires=[
        'transitions==0.6.9',
        'pygraphviz==1.5',
        'dill',
        'scikit-learn',
        'matplotlib',
        'numpy==1.16.4', 
        'Keras==2.2.4',
        'tensorflow==1.14.0',
    ],
    url='https://github.com/Musketeer-H2020/RobustMMLL'
)
