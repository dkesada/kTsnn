from setuptools import setup

setup(
    name='kTsnn',
    version='0.0.0.9000',
    packages=['kTsnn', 'kTsnn.src', 'kTsnn.src.nets'],
    url='https://github.com/dkesada/keras_TDNN',
    license='MIT',
    author='dkesada',
    author_email='dkesada@gmail.com',
    description='Implementation in keras of some neural networks related with time series short and long-term forecasting.',
    install_requires=['plotly', 'keras', 'pandas']
)
