# Beat-to-Beat Blood Pressure Estimation using Machine Learning

## Introduction
Cardiovascular disease is the cause of almost 50% of mortality due to non-communicable disease, in depth 45% of all deaths in Europe. Hypertension - high blood pressure (BP) - has been stated to be one of the highest primary risk factors for the majority of cardiovascular diseases. Coronary heart disease, fibrillation and strokes are referred back to hypertension. Additionally hypertension is an considerable risk factor for neurodegenerative diseases such as Alzheimer's disease, Parkinson's disease or dementia. According to the WHO, the cases of hypertension among adults aged 30-79 years have been immensely rising worldwide from 1990 to 2019. Despite the high risk radiated by hypertension and risk acceptance of the society, the diagnosis rate is calculated to be at only 46%. 

Therefore, ambulant BP monitoring or cNIBPM is of great interest to assess the physical health condition of the world's population in daily life along with a continuous and user friendly application. However, until this day, invasive methods, such as intra-arterial catheter are the goldstandard for \acrshort{BP} measurement, but are limited due to discomfort, risk of infections and serious patient injury in case of disruptive disconnection. Current non-invasive BP methods are cuff-based and discontinuous or measure only intermittent, whilst not being able to capture beat-to-beat characteristics of BP, such as sphygmomanometry. Common ambulatory BP measurements are often performed as volume clamp method or applanation tonometry over periods of 24 and 48 h. Although these measurements provide acceptable accuracy, the BP readings are questioned against their robustness and reliability. In addition, by using cuff-based applications they are often noisy, uncomfortable or disruptive for patients, in particular during sleep.

Recent papers describe methods for BP estimation using pulse wave velocity (PWV) as a surrogate approach. This concept is based on the assumption that the velocity of the arterial pulse traveling through an artery is affected by physiological variations of the vessel and patients, such as BP, age and the autonomous nervous system. Although the PWV is not directly measurable, it can be approximated by its surrogate, the pulse transit time (PTT) respectively. However numerous works tried to establish a BP estimation method using PTT, more and more studies use the pulse arrival time (PAT) instead, because of the slightly ease of measurement. Additionally, the number of surrogate approaches using Artificial Intelligence with focus on the photoplethysmography (PPG) has increased over years. Typically the input consists of extracted features or PPG signal excerpts and is fed into various types of models, such as random trees or forests over support vector machines, to deep neural networks. However, the data availability provides some limitations. Various works use private or subject-limited datasets, whereas on the other hand the dataset quality is poor. Therefore, the comparison of results in terms of reliability is unfair. 

This work aims to address this limitation by: i) extracting PPG features and absolute PAT values ii) training XGBoost models to predict PAT by the extracted features; and iii) using these features, along with either extracted PAT (ePAT) or predicted PAT (pPAT), to train various XGBoost models for BP estimation. Chapter \ref{sec:Background} states the theoretical background of PAT and its relationship with BP according to various works. The extraction of PPG-based features and the BP reference along with the extraction algorithm of PAT is introduced. At last, a quantitative evaluation and qualitative discussion of the resultant PAT and BP estimation is conducted based on the models' performances in relation to comparable publications. 

## Extraction of fiducial points using pyPPG
pyPPG is a standardised toolbox to analyze long-term finger PPG recordings in real-time. The toolbox extracts state-of-the-art PPG biomarkers (i.e. pulse wave features) from PPG signals. The algorithms implemented in the pyPPG toolbox have been validated on freely available PPG databases. Consequently, pyPPG offers robust and comprehensive assessment of clinically relevant biomarkers from continuous PPG signals.

The following steps are implemented in the pyPPG toolbox:

1. Loading a raw PPG signal: The toolbox can accept PPG signals in various file formats such as .mat, .txt, .csv, or .edf. These files should contain raw PPG data along with the corresponding sampling rate.
     + .mat: Data should be stored in two variables within the file: (i) ‘Fs’ representing the sampling frequency, and (ii) ‘Data’, a vector containing the raw PPG signal.
     + .txt: The raw PPG signal should be stored in tabular form (single tab or space-delimited), and you need to provide the sampling frequency as an input parameter to the script using ‘fs’.
     + .csv: This format stores raw PPG signal data with comma separation. Similar to .txt, the sampling frequency must be provided as an input parameter to the script using ‘fs’.
     + .edf: The European Data Format is supported, and it applies ‘Pleth’ channel by default. However, if using a different channel name, then the user needs to define it themselves.
2. Preprocessing: The raw PPG signal is filtered to remove noise and artifacts. Subsequently, the first, second, and third derivatives (PPG’, PPG’’, and PPG’”) of the PPG signal are computed and filtered. The resampling of the filtered PPG signal to 75 Hz is specifically performed for systolic peak detection.
3. Pulse wave segmentation: The toolbox employs a peak detector to identify the systolic peaks. It uses an improved version of a beat detection algorithm originally proposed in (Aboy et al. 2005). Based on the peak locations, the toolbox also detects the pulse onsets and offsets, which indicate the start and end of the PPG pulse waves.
4. Fiducial points identification: For each pulse wave, the toolbox detects a set of fiducial points.
5. Biomarker engineering: Based on the fiducial points, a set of 74 PPG digital biomarkers (i.e. pulse wave features) are calculated.

The pyPPG toolbox also provides an optional PPG signal quality index based on the Matlab implementation of the work by (Li et al. 2015).

![Alt text](./docs/pyPPG_Pipeline.svg)

The toolbox identifies individual pulse waves in a PPG signal by identifying systolic peaks (sp), and then identifying the pulse onset (on) and offset (off) on either side of each systolic peak which indicate the start and end of the pulse wave, respectively.

![Alt text](./docs/pyPPG_Sample.svg)

As example provided below, the pyPPG toolbox facilitates the loading of raw PPG signals from various file formats, including .mat, .csv, .txt, or .edf. This enables then the extraction of fiducial points from PPG signals, encompassing PPG, PPG’, PPG’’, and PPG’’’. Here, the extracted fiducial points are visually represented through plotting:

![Alt text](./docs/PPG_FiducialPoints.png)
