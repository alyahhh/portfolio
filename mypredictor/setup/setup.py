from setuptools import setup, find_packages

setup(
    name='network_behavior_analysis_ids',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'flask',
        'flask_cors',
        'joblib',
        'pandas',
        'pymongo',
        'scikit-learn'
        'numpy'
        'xgboost'
        'Flask-Caching'
        'google-api-python-client'
        'google-auth'
        'python-dotenv'
    ],
    entry_points={
        'console_scripts': [
            'predictor=flask_inital:main',  # Adjust this line as needed for your project
        ],
    },
)

