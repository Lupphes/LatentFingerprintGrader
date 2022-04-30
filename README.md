# LatFigGra – Latent Fingerprint Grader

LatFigGra is a Python library for grading fingerprints. It grades fingerprints based on the number of minutiae points, number of ridges, contrast, sinusoidal similarity and ridge thickness.

## Abstract
Dactyloscopic traces are one of the critical aspects of biometric identification. They rep-
resent an element by which people can be recognised and authorised. Nonetheless, it is
necessary to evaluate fingerprint value by the number of features it can provide and iden-
tify if it is usable or unusable and therefore tell us how much of a value it can bring. We
established a process that grades fingerprints based on contextual and statistical values by
using various enhancements and grading algorithms. These algorithms can determine if the
fingerprint is good quality and can be used for future processing or discarded. We divided
fingerprints into quality groups based on their quality of minutiae points, the number of
ridges, contrast, sinusoidal similarity and ridge thickness. We successfully evaluated fin-
gerprints and grouped them similarly to the NIST SD27 quality groups. The results from
algorithms allowed us to draw conclusions about graded fingerprints’ quality and rate their
usability.  

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
Put the model folder in the root directory, or do not forget to specify the path while running the program. You can download it from one of the following addresses [VUT Google Drive](https://drive.google.com/drive/folders/1QUJu4xwiIpOCbsu2zc5goLL8_UXD3ua3?usp=sharing) or [MSU prip-lab Google Drive](https://drive.google.com/drive/folders/1QUJu4xwiIpOCbsu2zc5goLL8_UXD3ua3?usp=sharing).

## Usage

```
usage: LatFigGra [-h] [-g GPU] [-e EXT] -s SDIR [-d DDIR] [-m MODELS] [-r]
                 [-p PPI]

LatFigGra (Latent Fingerprint Grader) 2022, Author: Ondřej Sloup (xsloup02)

optional arguments:
  -h, --help            show this help message and exit
  -g GPU, --gpu GPU     Comma-separated list of graphic cards which the script
                        will use for msu_afis. By default `0`.
  -e EXT, --ext EXT     File extension to which format the script will
                        generate the output images as. By default `jp2`.
  -s SDIR, --sdir SDIR  Path to the input folder, where the source images
                        should be.
  -d DDIR, --ddir DDIR  Path to the destination folder, where the script will
                        store fingerprint images and logs.
  -m MODELS, --models MODELS
                        Path to the model folder. By default `./models`.
  -r, --regenerate      Flag to regenerate already computed fingerprints
                        (their pickle files) despite their existence.
  -p PPI, --ppi PPI     PPI (Pixels per inch) under which the scanner scanned
                        the fingerprints. By default 500.
```
The basic starts need to specify the input folder with `-s`. It is recommended to specify PPI as it the script will yield better results and output folder.

## Bit for the opponent
I included everything needed on the attached SDHC card, especially the `models` folder, which contains the necessary models to run the 
enhancement process and the NIST SD27 latent dataset in the `SD27-lat` folder, where the graded fingerprints are. I also computed the
dataset in advance, as it takes around 2 hours to calculate it completely for all 292 fingerprints, and the result can be found in 
the `out` folder. 

The fingerprint grading process can be executed with this command inside the virtual environment:
```bash
python demo.py -s SD27-lat/ -d out/ --ppi 1000 
```
where `SD27-lat/` is the path to the dataset folder, `out/` is the output folder, and as the dataset was recorded in 1000 PPI, it is 
specified. The script can be run with the `-r` option with the same destination and source folder, regenerating everything in that folder 
and launching the enhancement process.

The test script can be run similarly with the command:
```bash
python test.py -s out/log.json -d out/
```
where the `-s` specify the source JSON file which will be analysed, and the `d` specify the destination. More information
can be found with the `--help`.

## Citations of code

[Original MSU Latent AFIS repository](https://github.com/prip-lab/MSU-LatentAFIS) for fingerprint enhancement.

[Edited MSU Latent AFIS repository](https://github.com/manuelaguadomtz/MSU-LatentAFIS) by Manuel Aguado Martinez with LogGabor filter

[Edited MSU Latent AFIS repository by me](https://github.com/Lupphes/MSU-LatentAFIS) to restructure the repository.

[Low-pass filter](https://github.com/guillaume-chevalier/filtering-stft-and-laplace-transform) for the extracted perpendicular line.

[Michelson's contrast](https://stackoverflow.com/questions/57256159/how-extract-contrast-level-of-a-photo-opencv) solution with OpenCV.


## Contributing
Pull requests, forks and other changes are welcomed are welcome.

## MD5 hash
MD5 hashes calculated for files can be found here:
[bachelor.lupp.es](https://bachelor.lupp.es/) – My personal site


## License
[GPLv3](https://choosealicense.com/licenses/gpl-3.0/)