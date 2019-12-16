from setuptools import setup
from setuptools import find_packages


setup(name='alana_learning_to_rank',
      version='1.0.0',
      description='Neural ranker for Alana, Heriot-Watt University\'s Alexa Prize Socialbot',
      author='Igor Shalyminov',
      author_email='ishalyminov@gmail.com',
      url='https://github.com/WattSocialBot/alana_learning_to_rank',
      download_url='https://github.com/WattSocialBot/alana_learning_to_rank.git',
      license='MIT',
      install_requires=['nltk==3.3',
                        'pandas==0.23.0',
                        'scikit-learn==0.19.1',
                        'tensorflow-gpu==1.15.0',
                        'configobj==5.0.6',
                        'ner==0.1'],
      packages=find_packages(),
      package_data={'alana_learning_to_rank': ['*']})
