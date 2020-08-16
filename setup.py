from setuptools import setup, find_packages

setup(name='time-blender',
      version='0.2.0',
      description='A compositional time series generator.',
      url='https://github.com/paulosalem/time-blender',
      author='Paulo Salem',
      author_email='paulosalem@paulosalem.com',
      license='MIT',
      packages=find_packages(),
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent"],
      include_package_data=True,
      install_requires=[
            'pandas', 'numpy', 'clize', 'hyperopt', 'sigtools', 'matplotlib', 'scikit-learn'
      ],
      scripts=['bin/time_blender'])