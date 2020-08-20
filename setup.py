from setuptools import setup, find_packages

setup(
    name='sentiment',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click >= 7.1.2',
        'scikit-learn >= 0.23.2',
        'joblib >= 0.16.0',
    ],
    entry_points='''
        [console_scripts]
        sentiment=sentiment.sentiment:sentiment
    '''
)

