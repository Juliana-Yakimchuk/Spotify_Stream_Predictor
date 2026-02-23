# Project Title: Spotify_Stream_Predictor

##💡 Project Overview
The project builds an ML Predictive Model to predict Spotify streams of a song based on audio features. The goal is to understand how musical characteristics relate to sustained listener engagement and to provide actionable insights for artists and labels.


## 🧐 Problem Statement
The initial goal of this project was to predict total Spotify stream counts using audio features. During exploratory analysis, it became clear that total streams are strongly confounded by exposure time (release date) and data availability for newer tracks.

To better align the target variable with the underlying data-generating process, the modeling objective was revised to predict **average monthly streams**, which reflects streaming intensity rather than cumulative exposure.

##🗂️ Dataset
- **Sources**: 
   1. Chartmetric track analytics report on extreme metal genres for 2005-2025 https://chartmetric.com/features/track-analytics
🔴_Note: Tracks released in 2025 were excluded from modeling due to incomplete feature availability and unstable stream counts._
   
   2.Webscraped data of audio features from Reccobeats for specific tracks (ISRCs) pulled previously from Chartmetric https://reccobeats.com/docs/apis/reccobeats-api
 
- **Description**:
    - Chartmetric report contains track information such as name of track and artist, release date, ISRC (International Standard Recording Code), stream counts and playlist save data from various popular music platforms.
    - Reccobeats data contains audio features for tracks from Chartmetric pulled from ISRCs obtained in Chartmetric report. Some records contain multiple versions of the same ISRC (e.g. album vs. single vs. remastered versions). The following audio features are present: 
    
|       Field      |  Type | Description                                           |
|:----------------:|:-----:|:----------------------------   ----------------------:|
| acousticness     | Float | Confidence (0.0 to 1.0) that the track is acoustic. Higher values indicate more natural sounds. |
| danceability     | Float | Suitability for dancing (0.0 to 1.0). Higher values indicate more rhythmically engaging tracks. |
| energy           | Float | Intensity and liveliness (0.0 to 1.0). Higher values indicate more energetic tracks.            |
| instrumentalness | Float | Likelihood of no vocals (0.0 to 1.0). Values above 0.5 suggest instrumental tracks.             |
| liveness         | Float | Probability of a live audience (0.0 to 1.0). Values above 0.8 strongly suggest a live track.    |
| loudness         | Float | Average loudness in decibels (dB). Typically ranges between -60 and 0 dB.                       |
| speechiness      | Float | Presence of spoken words (0.0 to 1.0). Values above 0.66 indicate mostly speech.                |
| tempo            | Float | Estimated tempo in beats per minute (BPM). Typically ranges between 0 and 250.                  |
| valence          | Float | Emotional tone (0.0 to 1.0). Higher values indicate a happier mood, lower values a sadder one.  |


## 🛠️ Approach & Methodology
1. **Data Cleaning & Preprocessing**:
   - Corrected data types
   - Transformed duplicate ISRCs into one data point using median of features
   - The target variable was transformed to average monthly streams to control for stream accumulation over the years and subsequently log-transformed to reduce right-skewness and stabilize variance, improving the suitability of linear modeling assumptions. 

2. **Exploratory Data Analysis (EDA)**:
   - Comparison of fetched vs. missing ISRCs
   - Distribution of monthly streams
   - Distribution of features
   - Calculate Pearson correlation of features against the target
 
3. **Feature Engineering**:
Exploratory analysis was conducted to assess whether nonlinear transformations or interaction terms could meaningfully strengthen the relationship between audio features and the log-transformed target variable (average monthly streams).The following engineered features were explored:

 - Loudness squared (to capture potential curvature)
 - Centered loudness squared
 - Interaction term between energy and danceability

Scatterplots of these engineered variables against the log target were examined and compared to the original features.

Visual inspection of scatterplots suggested no meaningful reduction in variance or clearer functional relationship relative to the original predictors. In several cases, dispersion appeared similar or greater than that of the untransformed predictors, suggesting limited additional explanatory signal. For that reason, no adidtional engineered features were added to the dataset. We also extracted the year of release from the release date.

Release year was extracted from the full release date to capture temporal effects related to platform growth and industry dynamics. Visual inspection indicated a clearer association with the target compared to many audio features, and it was therefore retained for modeling.

4. **Modeling**:
   - Baseline linear regression
   - Tree-based models (Random Forest / XGBoost)
   - ANN

5. **Model Explainability (SHAP Values)**:

6. **Deployment**:

## 📊 Results & Metrics
| Metric            | Score |
|-------------------|-------|
| RMSE              |  TBD  |
| Adjusted R²       |  TBD  |


- **Top Features Influencing Predictions**:  
  

## 🔍 SHAP Visualizations
(To be completed after model training)

## 🌐 Deployment Link


## 📝 Business Impact
This model provides insights into how audio features affect streaming preformance to help artists and labels guide business decisions, such as:
1. Invest into a high quality artwork
2. Pitch the track to Spotify playlist curators
3. Shoot a music video
4. Drop exlusive song-themed merchandise 
5. Create additional social media content
6. Run ads and promos on social media


## 📝 Limitations & Assumptions
   - Streaming counts are influenced by marketing, artist popularity, and platform dynamics not captured in audio features
   - The model focuses on association rather than causation
   - Results may not generalize beyond the genres represented in the dataset

## 🔮 Future Work: Audio-Derived Feature Extraction

To support true pre-release inference, future iterations of this project could replace third-party audio features with features extracted directly from raw audio files using digital signal processing (DSP) libraries such as Librosa or Essentia. This would enable:
   - Feature computation before public release
   - Independence from platform-specific APIs
   - Greater reproducibility and long-term maintainability

Due to time and resource constraints, this step is outside the scope of the current implementation but is a natural extension of the existing pipeline.

## 📂 Project Structure
```
Spotify_Stream_Predictor/
├─ data/
│  ├─ raw/ 
      └─ chartmetric_raw.csv           # raw data
│  │  └─ reccobeats_raw.parquet  # LOCAL ONLY
│  └─ metadata/          # progress tracking files
│     └─ fetched_isrcs.txt       # LOCAL ONLY
├─ notebooks/             # exploratory notebooks
├─ src/                   # fetching, cleaning, processing scripts
├─ README.md
└─ shap_summary_plot.png   # SHAP Summary Plot
└─ model.pkl               # Serialized Model
```

## 🚀 How to Run Locally
```bash
# Clone the repo
git clone https://github.com/Juliana-Yakimchuk/Spotify_Stream_Predictor.git

# Install dependencies
pip install -r requirements.txt

# Run the app

```

## 🧰 Tech Stack & Tools
- Python (Pandas, Scikit-learn)
- SHAP (Explainable AI)
- Streamlit
- Git & GitHub

## 📬 Contact
For queries or collaborations, feel free to connect with me on [LinkedIn](https://linkedin.com/in/juliana-yakimchuk).
