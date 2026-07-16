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

---

## 5. Algorithmic Safeguards & Mitigations

To prevent automated diagnostic misinterpretation and ensure safe usage, VitaVoice incorporates two major software safeguards:

### A. Pre-Inference Recording Quality Control
Low-quality audio (e.g., quiet whisper, loud background noise, digital clipping) leads to garbage-in, garbage-out model behavior. 
- **SNR and Noise Assessment**: Standardizes sample analysis. Audio with high noise percentage (>50%) or low SNR (<10 dB) is flagged as "Noisy".
- **Clipping Detection**: Alerts when the signal exceeds a critical threshold ($\ge 0.99$ peak amplitude), indicating digital distortion.
- **Suitability Filtering**: Assigns a 1-5 star quality score. Any audio rated $\le 2$ stars generates a clear warning label instructing the user that results are less reliable due to recording conditions.

### B. Post-Inference Response Enrichment
Raw prediction probabilities can be confusing or alarming. The backend enriches raw classification outputs into clear, safe clinical contexts:
- **Certainty Calibration**: Reports margin-based confidence tiers ("Very High", "High", "Moderate", "Low") to signal when a user's voice is close to the decision boundary (high variance/low certainty) or highly defined.
- **Risk-Stratified Actionable Recommendations**: Generates tailored clinical suggestions based on risk. Low risk recommends wellness tracking; moderate risk recommends retesting in quiet rooms to eliminate transient acoustic biases; high risk recommends formal ENT/Neurological diagnostic assessments.
- **Explainable AI (SHAP) & Natural Language**: Automatically synthesizes complex SHAP vector arrays into a plain-English explanation (e.g., "screening result was primarily influenced by increased vocal jitter, which may indicate minor speech irregularities...") to demystify "black-box" predictions.
- **Direct Disclaimers**: Ensures the generated PDF reports and interactive frontend feature explicit responsible AI disclaimers clarifying that the tool is a wellness screening aid, not a diagnostic platform.
