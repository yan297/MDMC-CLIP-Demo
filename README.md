# CLIP Zero-Shot Classification Demo

This repository contains a demo for zero-shot image classification using a fine-tuned CLIP model, as part of a research paper. The demo evaluates the model on two datasets: `Test_new_BCMD` and `Test_new_CMMD`.

## Project Structure
````
CLIP-ZeroShot-Demo/
├── data/                # Place downloaded datasets here
├── model/               # Place downloaded model parameters here
├── notebooks/           # Jupyter Notebook for interactive demo
│   └── demo.ipynb
├── demo.py              # Main script for quick execution
├── requirements.txt     # Python dependencies
└── README.md            # This file
````

## Prerequisites
- Python 3.8 or higher
- GPU (optional, for faster inference)
- Jupyter Notebook (optional, for running the interactive demo)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yan297/MDMC-CLIP-Demo.git
cd MDMC-CLIP-Demo
```

### 2. Create and Activate a Virtual Environment
#### On Windows
```bash
python -m venv MDMC-CLIP-Demo
MDMC-CLIP-Demo\Scripts\activate
```
#### On macOS and Linux
```bash
python3 -m venv MDMC-CLIP-Demo
source MDMC-CLIP-Demo/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Download Resources
- Download the `Test_BCMD` and `Test_CMMD` datasets from the following links:
  - [Test_BCMD](https://drive.google.com/drive/folders/1A5iL6PWJsvm_aXfwQApwbyaM5zt595zs?usp=drive_link)
  - [Test_CMMD](https://drive.google.com/drive/folders/1b9FgLTt6jMatr_jPOnp9JKScyQJc-yCC?usp=drive_link)

- Download the fine-tuned model parameters from the following link:
  - [Model Parameters](https://drive.google.com/drive/folders/1XrAF-4Hqd3LFE4wERDVxBWTfFtMaTS5q?usp=drive_link)

After downloading, your directory should look like: 
````
CLIP-ZeroShot-Demo/
├── data/
│   ├── Test_BCMD/
│   └── Test_CMMD/
├── model/
│   └── L-OpenCLIP-4e-6/
├── notebooks/
│   └── demo.ipynb
├── demo.py
├── requirements.txt
└── README.md
````
### 5. Run the Demo
#### Run the Script Directly: 
```bash
python demo.py
```
#### The script will evaluate the model on both datasets and print the accuracy to the terminal.

