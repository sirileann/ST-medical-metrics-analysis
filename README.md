# medical-metrics-analysis
medical-metrics-analysis, is the "ECG and PPG Analysis" Semester Project from Siri Rüegg 

## Get started:
### Installation Guide:

1. Clone medical-metrics-analysis repository
```bash
git clone git@github.com:SCAI-Lab/medical-metrics-analysis.git
```
2. Create virtualenv environment and activate it. 
```bash 
cd medical-metrics-analysis
python3.8 -m venv venv
source venv/bin/activate
```
3. Install requirements
```bash
pip install -r requirements.txt
```

## Toolbox
### ADL_quality_ECG.ipynb
This python notebook contains the workflow for assessing the signal quality during various activities of daily living for ECG signals. 

### ADL_quality_PPG.ipynb
This python notebook contains the workflow for assessing the signal quality during various activities of daily living for PPG signals. Additionally, there are extra functions listed to compare the results from the ECG and PPG analysis. The required .csv files are listed in ..\examples.

### hr_ppg_ecg_correlation.py
This script can be used to calculate the correlation between the heart rate and the pulse rate by calculating the L2-norm between those two signals. 

### signal_utils.py
This script contains all helper functions used in the ADL_quality scripts. 

## Data
If required, the data used for the analysis in the semester thesis "Personalised Signal Quality Assessment for ECG and PPG Signals - Towards Quantifying Cardiovascular Output in Activities of Daily Living for the SCI Population" of Siri Rüegg can be downloaded from this folder. 
