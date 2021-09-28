from setuptools import setup, find_packages
 
classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: Microsoft :: Windows',
  'Operating System :: POSIX :: Linux',
  'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='fingerprintqualitygrader',
  version='0.0.1',
  description='Grades a quality of fingerprints',
  long_description=open('./../README.md').read() + '\n\n' + open('./../CHANGELOG.txt').read(),
  url='',  
  author='Ondřej Sloup',
  author_email='ondrej.sloup@protonmail.com',
  license='GPLv3', 
  classifiers=classifiers,
  keywords='fingerprints', 
  packages=find_packages(),
  install_requires=[''] 
)