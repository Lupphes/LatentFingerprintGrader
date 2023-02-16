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
    name='latfiggra',
    version='1.0.0',
    description='Algorithmic Evaluation of the Quality of Dactyloscopic Traces',
    long_description=open('README.md').read() +
    '\n\n' + open('CHANGELOG.txt').read(),
    url='',
    author='Ondřej Sloup',
    author_email='ondrej.sloup@pm.me',
    license='GPLv3',
    classifiers=classifiers,
    keywords='fingerprints, algorithmic, evaluation, quality, dactyloscopic, traces',
    packages=find_packages(),
    install_requires=[
        "absl-py == 1.0.0",
        "astor == 0.8.1",
        "cached-property == 1.5.2",
        "cachetools == 4.2.4",
        "certifi == 2021.10.8",
        "charset-normalizer == 2.0.12",
        "cycler == 0.11.0",
        "fonttools == 4.31.2",
        "gast == 0.2.2",
        "google-auth == 1.35.0",
        "google-auth-oauthlib == 0.4.6",
        "google-pasta == 0.2.0",
        "grpcio == 1.44.0",
        "h5py == 3.6.0",
        "idna == 3.3",
        "imageio == 2.16.1",
        "importlib-metadata == 4.11.3",
        "Keras-Applications == 1.0.8",
        "Keras-Preprocessing == 1.1.2",
        "kiwisolver == 1.4.0",
        "Markdown == 3.3.6",
        "matplotlib == 3.5.1",
        "networkx == 2.6.3",
        "numpy == 1.21.5",
        "oauthlib == 3.2.0",
        "opencv-contrib-python == 4.5.5.64",
        "opt-einsum == 3.3.0",
        "packaging == 21.3",
        "pandas == 1.3.5",
        "Pillow == 9.0.1",
        "protobuf == 3.19.4",
        "pyasn1 == 0.4.8",
        "pyasn1-modules == 0.2.8",
        "pyparsing == 3.0.7",
        "python-dateutil == 2.8.2",
        "pytz == 2022.1",
        "PyWavelets == 1.3.0",
        "requests == 2.27.1",
        "requests-oauthlib == 1.3.1",
        "rsa == 4.8",
        "scikit-image == 0.19.2",
        "scipy == 1.7.3",
        "six == 1.16.0",
        "tensorboard == 2.0.2",
        "tensorflow == 2.0.0",
        "tensorflow-estimator == 2.0.1",
        "termcolor == 1.1.0",
        "tifffile == 2021.11.2",
        "typing_extensions == 4.1.1",
        "urllib3 == 1.26.9",
        "Werkzeug == 2.2.3",
        "wrapt == 1.14.0",
        "zipp == 3.7.0"
    ],
)
