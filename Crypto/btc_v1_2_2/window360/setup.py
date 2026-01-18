from setuptools import setup, find_packages

setup(
    name='fina',
    version='0.1.0',
    description='A Python package for financial trading strategies',
    author='When2buy',
    author_email='when2buy@aitist.ai',
    packages=find_packages(include=['fina', 'fina.*']),
    install_requires=[
        'pandas>=2.2.1',
        'numpy>=1.26.4',
        'python-dotenv>=1.0.1',
        'yfinance>=0.2.37',
        'schedule>=1.2.1',
        'loguru>=0.7.2',
        'pytz>=2024.1',
        'requests>=2.31.0',
        'alpaca-py'
    ],
    python_requires='>=3.8',
) 