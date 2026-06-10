# Responsible AI & Clinical Limitations

## 1. Intended Use and Scope
**VitaVoice is an AI-assisted voice pathology screening aid, not a diagnostic system.**

It is designed to analyze vocal dysphonia (vocal instability) from sustained vowel phonation samples. It serves as a rapid, low-cost, and non-invasive **pre-clinical screening tool** to identify voice biomarkers associated with chronic conditions (such as Parkinson's disease). 

### What it is:
* An educational and research demonstration.
* A screening aid to highlight vocal tremors, hoarseness, and pitch instabilities.
* A tool to encourage early-stage clinical consultation with primary care physicians, Speech-Language Pathologists (SLPs), or Otolaryngologists (ENTs).

### What it is NOT:
* A medical diagnostic device. It has not been cleared by the FDA or any regulatory body.
* A replacement for clinical larynx visualization, motor exams, or formal neurological evaluations.
* A definitive test for Parkinson's disease.

---

## 2. Demographic and Acoustic Biases

Like all machine learning models trained on voice datasets, VitaVoice is subject to systemic biases. Researchers and developers must account for:

### A. Microphone and Hardware Bias
* **The Bias:** Different microphones (phone built-in mic, studio condenser, laptop headset) have varying frequency response curves, noise floors, and sampling characteristics. A cheap microphone with high self-noise can artificially lower the Harmonics-to-Noise Ratio (HNR) and inflate Jitter calculations, leading to **false positives**.
* **Mitigation:** VitaVoice implements **Loudness Normalization** (scaling the peak signal) and **Spectral Gating Noise Reduction** to isolate and subtract steady background hums. The frontend also features a **Microphone Clipping Detector** to prevent digital distortion.

### B. Demographic Bias (Age, Gender, Accent)
* **The Bias:** Voice fundamental frequency (F0) is heavily split by physiological gender (typically 85–155Hz for adult males, 165–255Hz for adult females). Age-related larynx calcification also naturally increases pitch jitter. If the training data lacks gender balance or representative age groups, the classification boundary will skew.
* **Mitigation:** Feature scaling (`StandardScaler`) standardizes inputs across the cohort. Clinical feature thresholds on the dashboard show gender-appropriate ranges.

### C. Language and Accents
* **The Bias:** The models are evaluated on sustained vowel phonations ("ah" / "oh") which are physically universal. However, different phonetic traditions and dialects can slightly alter larynx shape during vowels.
* **Mitigation:** The pipeline uses only voiced segments extracted by the **Voice Activity Detector (VAD)** to keep features restricted to vowel phonations.

---

## 3. Training Data Limitations
* **Cohort Size:** The default model checkpoint is trained on the **Oxford Parkinson's Disease Detection Dataset** consisting of only **195 recordings from 31 subjects**. This is a small research sample.
* **Lack of Diversified Baseline:** The dataset represents a narrow geographic and socio-demographic cohort. Results should not be generalized to larger clinical populations without local calibration.
* **Swappable Dataset Layer:** To address this limitation, the backend architecture has been updated with a unified **Dataset Abstraction Layer (`BaseDatasetLoader`)** so that clinical research groups can easily swap in larger, multi-ethnic clinical datasets (such as the Spanish PC-GITA or custom folders of real patient WAVs) to retrain and validate the models.

---

## 4. Explaining the Biomarkers

VitaVoice prioritizes explainable AI by mapping model predictions directly to physical biomarkers on the dashboard:

| Biomarker | Physical Source | Parkinsonian Manifestation |
| :--- | :--- | :--- |
| **Pitch Jitter** | Vocal fold vibration frequency instability | Elevated. Parkinson's affects the laryngeal muscle control, resulting in micro-tremors during vocal cord closure. |
| **Amplitude Shimmer** | Vocal fold vibration loudness instability | Elevated. Reduced subglottal pressure control makes it difficult to maintain a steady vocal volume. |
| **HNR (Harmonics-to-Noise)** | Completeness of vocal fold closure | Lowered. Incomplete closure leads to turbulent air leakage, causing a breathy, hoarse, or noisy voice. |
| **MFCCs** | Shape of the vocal tract filter | Altered. Reduced movement range of the tongue, lips, and jaw changes the spectral envelope shape. |
