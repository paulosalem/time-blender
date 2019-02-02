from setuptools import setup

setup(name='time-blender',
      version='0.1.2',
      description='A compositional time series generator.',
      url='https://github.com/paulosalem/time-blender',
      author='Paulo Salem',
      author_email='paulosalem@paulosalem.com',
      license='MIT',
      packages=['time_blender'],
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent"],
      include_package_data=True,
      install_requires=[
            'pandas', 'numpy', 'clize', 'pymc3', 'sigtools', 'matplotlib'
      ],
      scripts=['bin/time_blender'])