# Docked Compound Cluster Visualiser #
This application provides a GUI for visualising and clustering docked compounds based on molecular descriptors and fingerprints using DBSCAN. It includes a 3D viewer for selected molecules.

## Executable ##
### <ins>Requirement</ins> ###
A valid OpenEye license.

### <ins>Steps to run the executable</ins> ###
1. Set the environment variable:
```bash
%SystemRoot%\System32\setx OE_LICENSE "C:\path\to\oe_license.txt"
```
2. Double-click the executable.

## Script ##
### <ins>Python version</ins> ###
Python 3.9

### <ins>Dependencies</ins> ###
OpenEye Python Toolkit is not included here because it must be downloaded manually from https://openeye.app.box.com/s/ebywkngy0p45enmn3za21kb7dzf1mzbl/folder/306164025307, and requires a valid license.
A few Python packages need to be installed.

### <ins>Steps to run the script</ins> ###
1. Create a new virtual environment:
```bash
python -m venv env
env\Scripts\activate
```
2. Install the wheel:
```bash
pip install "C:\path\to\OpenEye_toolkits-<version>-py39-none-<platform>.whl"
```
3. Set the environment variable:
```bash
%SystemRoot%\System32\setx OE_LICENSE "C:\path\to\oe_license.txt"
```
4. Install the required packages:
```bash
pip install -r "C:\path\to\requirements.txt"
```
5. If requirements.txt is not working, install the packages manually:
```bash
pip install kneed matplotlib "numpy<2" pandas "pillow>=10,<11" py3Dmol rdkit-pypi scikit-learn ttkbootstrap
```
6. Run the script:
```bash
python "C:\path\to\ClusteringApp.py"
```
