# LatFigGra – Latent Fingerprint Grader

LatFigGra is a Python library for grading fingerprints. It grades fingerprints based on the number of minutiae points, number of ridges, contrast, sinusoidal similarity and ridge thickness.

## Abstract
Dactyloscopic traces are one of the critical aspects of biometric identification. They represent an element by which people can be authenticated and authorised. Nonetheless, it is necessary to evaluate if a given fingerprint is valid by the number of features it provides and decide if it is usable or useless. This analysis of features tells us how valuable the fingerprint is. We established a process that grades fingerprints based on contextual and statistical values using various enhancements and grading algorithms. These algorithms can determine if the fingerprint is good quality and whether it can be used for future processing or should be discarded. We divided fingerprints into groups based on the quality of their minutiae points, number of ridges, contrast, sinusoidal similarity and ridge thickness. We successfully evaluated fingerprints and grouped them similarly to the grouping in the NIST SD27 dataset. The algorithm's results allowed us to draw conclusions about graded fingerprints' quality and rate their usability.

## Installation 

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages to run the library demo.

Create a virtual environment:
```bash
virtualenv --python=$(which python3.7) venv
```
Enter the virtual environment:
```bash
source venv/bin/activate
```
Install the required packages:
```bash
pip3 install -r requirements.txt
```
Put the model folder in the root directory, or do not forget to specify the path of the model folder while running the program. 
You can download it from one of the following addresses [VUT Google Drive](https://drive.google.com/drive/folders/1QUJu4xwiIpOCbsu2zc5goLL8_UXD3ua3?usp=sharing), 
[MSU prip-lab Google Drive](https://drive.google.com/drive/folders/12wgcb5K_T5yNTsLYb557JLdhf2IHvo_j?usp=sharing), or 
[VUT FIT Next Cloud](https://nextcloud.fit.vutbr.cz/s/CzAckPWKqHrJwoe) (models.zip). 
This step should not be needed if you have an SDHC card with the prepared folder.

## Usage

```
usage: LatFigGra [-h] [-g GPU] [-e EXT] -s SDIR [-d DDIR] [-m MODELS] [-r]
                 [-p PPI]

LatFigGra (Latent Fingerprint Grader) 2022, Author: Ondřej Sloup (xsloup02)

optional arguments:
  -h, --help            show this help message and exit
  -g GPU, --gpu GPU     Comma-separated list of graphic cards which the script
                        will use for msu_afis. By default, `0`.
  -e EXT, --ext EXT     File extension to which format the script will
                        generate the output images. By default, `jp2`.
  -s SDIR, --sdir SDIR  Path to the input folder, where the source images
                        should be.
  -d DDIR, --ddir DDIR  Path to the destination folder, where the script will
                        store fingerprint images and logs.
  -m MODELS, --models MODELS
                        Path to the model folder. By default, `./models`.
  -r, --regenerate      Flag to regenerate already computed fingerprints
                        (their pickle files) despite their existence.
  -p PPI, --ppi PPI     PPI (Pixels per inch) under which the scanner scanned
                        the fingerprints. By default, 500.
```
For the most straightforward start, the demo scripts need specification of the input folder with `-s`. It is recommended to specify PPI as the script will yield better results.

## Part for the opponent
I included everything needed on the attached SDHC card and created the `LatentFingerprintGrader-2.0.2/` folder, which has everything prepared for execution. The folder contains the `models/` folder, which has all necessary models for the enhancement process and the NIST SD27 latent dataset in 
the `SD27-lat/` folder, where the graded fingerprints are. I also computed the dataset in advance, as it takes around 2 hours to calculate it 
entirely for all 292 fingerprints, and the result can be found in the `out/` folder. You can set it up on your own from the attached zip files or just run the command from the specified directory. The SDHC card is locked for writing, so it is needed that to copy the  `LatentFingerprintGrader-2.0.2/` folder to your computer.

The fingerprint grading process can be executed with this command inside the virtual environment:
```bash
python demo.py -s SD27-lat/ -d out/ --ppi 1000 
```
where `SD27-lat/` is the path to the dataset folder, `out/` is the output folder. The NIST SD27 dataset was scanned with PPI 1000; therefore, `--ppi 1000` is specified in the command. The script can be run with the `-r` option, which indicates that everything will be regenerated, even python pickle files; therefore, it also lanches the enhancement part of the algorithm, which takes additional time. Note that it is needed to specify the same destination and source folder if the `-r` command is not set to launch the grading process without enhancement.

The test script can be run similarly with the command:
```bash
python test.py -s out/log.json -d out/
```
where the `-s` specify the source JSON file which will be analysed, and the `d` specify the destination. More information
can be found with the `--help`.

## Requirements
The demo.py script needs at least 13GB, recommended 16GB, of RAM. It will load the Tensorflow charts into memory and enhance the fingerprints by them. This requirement should not be needed when running the demo script for grading using generated Pickle files.

## Citations of code

[Original MSU Latent AFIS repository](https://github.com/prip-lab/MSU-LatentAFIS) for fingerprint enhancement.

[Edited MSU Latent AFIS repository](https://github.com/manuelaguadomtz/MSU-LatentAFIS) by Manuel Aguado Martinez with LogGabor filter

[Edited MSU Latent AFIS repository by me](https://github.com/Lupphes/MSU-LatentAFIS) to restructure the repository.

[Low-pass filter](https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform) for the extracted perpendicular line.

[Michelson's contrast](https://stackoverflow.com/questions/57256159/how-extract-contrast-level-of-a-photo-opencv) solution with OpenCV.


## Contributing
Pull requests, forks and other changes are welcomed.

## MD5 hash
MD5 hashes calculated for files can be found here:
[bachelor.lupp.es](https://bachelor.lupp.es/) – My personal site
You can check the HASH with your favourite tool and compare it to the website.


## License
[GPLv3](https://choosealicense.com/licenses/gpl-3.0/)