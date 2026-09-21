# Determinanții participării femeilor pe piața muncii în economiile europene, 2001–2023

<p align="center">
  <img src="https://raw.githubusercontent.com/adda987/Participarea-femeilor-pe-piata-muncii/main/female_labor_participation/assets/logo_ads.png" alt="Project logo" width="140" />
</p>

<p align="center">
  <a href="https://participarea-femeilor-pe-piata-muncii-hntenjayuttsjzggt7hewa.streamlit.app/">
    <img alt="Live app" src="https://img.shields.io/badge/Live%20App-Open-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="R" src="https://img.shields.io/badge/R-4.x-276DC3?style=for-the-badge&logo=r&logoColor=white" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

Aplicație multipagină Streamlit dedicată analizei determinantelor participării femeilor pe piața muncii în economiile europene, în perioada 2001–2023.

## Despre proiect

Proiectul combină cercetarea academică cu o prezentare interactivă, explorând modul în care factori macroeconomici, demografici, educaționali, structurali, digitali și instituționali se asociază cu rata participării femeilor la forța de muncă.

Dashboard-ul include:

- revizuirea literaturii de specialitate
- dicționar interactiv de variabile
- analiză descriptivă și comparativă între țări
- econometrie panel și serii de timp
- modele de Machine Learning
- concluzii și implicații de politică publică

## Aplicația live

Accesați aplicația aici:

- https://participarea-femeilor-pe-piata-muncii-hntenjayuttsjzggt7hewa.streamlit.app/

## Structura proiectului

```text
female_labor_participation/
├── app.py
├── README.md
├── requirements.txt
├── assets/
│   ├── styles.css
│   └── logo_ads.png
├── components/
│   ├── sidebar.py
│   ├── html.py
│   ├── metric_cards.py
│   ├── charts.py
│   ├── tables.py
│   ├── article_cards.py
│   ├── empty_states.py
│   └── footer.py
├── config/
│   └── europe_countries.csv
├── data/
│   ├── P_Data_Extract_From_World_Development_Indicators.xlsx
│   ├── geo/
│   └── ...
├── pages/
│   ├── 01_Literature_Review.py
│   ├── 02_Data_and_Variables.py
│   ├── 03_Descriptive_Analysis.py
│   ├── 04_Cross_Sectional_Analysis.py
│   ├── 06_Time_Dynamics.py
│   ├── 07_Machine_Learning.py
│   └── 08_Results_and_Conclusions.py
├── R/
│   ├── analiza_transversala.R
│   ├── analiza_panel.R
│   └── analiza_serii_timp.R
├── src/
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── data_cleaning.py
│   ├── descriptive_statistics.py
│   ├── cross_sectional_models.py
│   ├── panel_models.py
│   ├── time_series.py
│   ├── machine_learning.py
│   └── utils.py
├── outputs/
│   ├── cross_sectional/
│   ├── machine_learning/
│   └── ...
├── literature/
│   ├── literature_matrix.xlsx
│   └── article_notes/
└── .gitignore
```

## Întrebarea de cercetare

Care sunt principalii factori macroeconomici, demografici, educaționali, structurali, digitali, instituționali și de politică familială asociați cu participarea femeilor pe piața muncii în economiile europene?

## Metodologie

Aplicația este structurată pe mai multe etape:

1. literatură și definiție conceptuală
2. colectare și validare a datelor
3. analize descriptive și comparări transversale
4. modele econometrice panel
5. dinamică temporală și șocuri economice
6. evaluare Machine Learning
7. interpretare și concluzii

## Tehnologii

- Python 3.10+
- Streamlit
- Pandas, NumPy, SciPy
- Plotly
- scikit-learn
- GeoPandas
- OpenPyXL
- XGBoost, SHAP
- R pentru analiza econometrică

## Pornire locală

Din directorul proiectului:

```bash
cd female_labor_participation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Analiză în R

Unele scene ale aplicației rulează calcule econometrice în R. Pachetele necesare pot fi instalate o singură dată:

```r
install.packages(c(
  "readxl", "dplyr", "tidyr", "ggplot2", "car", "lmtest",
  "tseries", "nortest", "moments", "caret", "glmnet",
  "MLmetrics", "broom", "sandwich", "jsonlite", "corrplot",
  "patchwork", "scales", "plm", "ggrepel", "forecast",
  "urca", "zoo", "vars"
))
```

Pentru a executa scripturile direct:

```bash
Rscript R/analiza_transversala.R
Rscript R/analiza_panel.R
Rscript R/analiza_serii_timp.R
```

## Date și surse

Datele sunt centrate pe indicatori din World Development Indicators și pe variabile de tip macroeconomic, structural și instituțional. În aplicație apar, printre altele:

- PIB pe locuitor
- inflație
- șomaj feminin
- fertilitate
- urbanizare
- educație terțiară
- utilizarea internetului
- controlul corupției

## Autor

Andreea-Daniela Sfetcu

## Observații

Acest proiect este un instrument de cercetare academică și de explorare bazată pe date. El are rolul de a sintetiza, explica și vizualiza modele economice relevante, fără a substitui analiza critică și interpretarea contextuală.
