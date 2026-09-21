# Determinanții participării femeilor pe piața muncii în economiile europene, 2001–2023

Aplicație multipagină Streamlit pentru un proiect academic și de știința datelor dedicat analizei factorilor asociați cu participarea femeilor pe piața muncii în economiile europene, în perioada 2001–2023.

Proiect dezvoltat de **Andreea-Daniela Sfetcu**.

## Întrebarea de cercetare

Care sunt principalii factori macroeconomici, demografici, educaționali, structurali, digitali, instituționali și de politică familială asociați cu participarea femeilor pe piața muncii în economiile europene în perioada 2001–2023?

## Obiective

- Organizarea literaturii de specialitate într-o matrice documentată.
- Construirea și validarea unei baze de date panel pentru economiile europene.
- Documentarea variabilelor într-un dicționar interactiv al datelor.
- Analiza statistică descriptivă, transversală, panel și temporală.
- Pregătirea unui cadru de Machine Learning cu validare atentă și prevenirea scurgerii de informație între antrenare și testare.
- Interpretarea economică a rezultatelor și formularea implicațiilor de politică publică.
- Documentarea parcursului personal al cercetării.

## Structura proiectului

```text
female_labor_participation/
├── app.py
├── pages/
│   ├── 01_Literature_Review.py
│   ├── 02_Data_and_Variables.py
│   ├── 03_Descriptive_Analysis.py
│   ├── 04_Cross_Sectional_Analysis.py
│   ├── 05_Panel_Econometrics.py
│   ├── 06_Time_Dynamics.py
│   ├── 07_Machine_Learning.py
│   └── 08_Results_and_Conclusions.py
├── components/
│   ├── sidebar.py
│   ├── metric_cards.py
│   ├── charts.py
│   ├── tables.py
│   ├── article_cards.py
│   ├── empty_states.py
│   └── footer.py
├── data/
│   ├── raw/
│   ├── intermediate/
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── data_cleaning.py
│   ├── missing_values.py
│   ├── descriptive_statistics.py
│   ├── cross_sectional_models.py
│   ├── panel_models.py
│   ├── time_series.py
│   ├── machine_learning.py
│   └── utils.py
├── assets/
│   ├── styles.css
│   └── logo_ads.png
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── reports/
├── literature/
│   ├── literature_matrix.xlsx
│   └── article_notes/
├── requirements.txt
├── README.md
└── .gitignore
```

## Instalare

Din folderul care conține proiectul:

```bash
cd female_labor_participation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Pornirea aplicației

```bash
streamlit run app.py
```

Aplicația citește automat fișierul Excel existent în folderul `data`.

## Dependențe R pentru analiza transversală

Pagina „Analiza transversală” rulează calculele statistice prin `Rscript`, folosind scriptul `R/analiza_transversala.R`. Pachetele R necesare trebuie instalate manual o singură dată:

```r
install.packages(c(
  "readxl", "dplyr", "tidyr", "ggplot2", "car", "lmtest",
  "tseries", "nortest", "moments", "caret", "glmnet",
  "MLmetrics", "broom", "sandwich", "jsonlite", "corrplot",
  "patchwork", "scales", "plm", "ggrepel", "forecast",
  "urca", "zoo", "vars"
))
```

Aplicația nu execută automat `install.packages()`. Analiza poate fi rulată direct din terminal:

```bash
Rscript R/analiza_transversala.R
```

Pentru pagina „Analiză econometrică panel”, analiza poate fi rulată direct prin:

```bash
Rscript R/analiza_panel.R
```

Pentru pagina „Serii de timp și dinamică temporală”, analiza poate fi rulată direct prin:

```bash
Rscript R/analiza_serii_timp.R
```

## Fișiere de date

- `data/P_Data_Extract_From_World_Development_Indicators.xlsx`: baza WDI utilizată în aplicație.
- `data/raw/`: fișiere brute descărcate din sursele originale, dacă vor fi păstrate separat.
- `data/intermediate/`: fișiere transformate intermediar.
- `data/dataset_path.txt`: cale relativă opțională către fișierul principal de date.
- `literature/literature_matrix.xlsx`: structură auxiliară; pagina de literatură folosește introducere manuală în aplicație.

## Surse de date planificate

- Eurostat: piața muncii, demografie, educație, digitalizare.
- Banca Mondială / WDI: indicatori macroeconomici și structurali.
- OCDE: politici familiale, fiscalitate, indicatori instituționali.
- OIM: indicatori comparabili ai ocupării și participării.
- Indicatorii de guvernanță mondială: guvernanță și controlul corupției.

Sursele, codurile indicatorilor, definițiile, unitățile de măsură și transformările vor fi documentate în dicționarul datelor după selectarea finală a variabilelor.

## Parcurs de dezvoltare

- Versiunea 0.1: schelet multipagină, identitate vizuală și componente reutilizabile, fără rezultate fictive.
- Versiunea 0.2: introducerea manuală și documentarea articolelor.
- Versiunea 0.3: integrarea setului de date și raport complet de calitate a datelor.
- Versiunea 0.4: analiză descriptivă și grafice exploratorii.
- Versiunea 0.5: analiză transversală și diagnostice.
- Versiunea 0.6: modele panel și robustețe.
- Versiunea 0.7: dinamică temporală și șocuri.
- Versiunea 0.8: Machine Learning și interpretabilitate.
- Versiunea 1.0: rezultate finale, concluzii, exporturi și raport.

## Stadiu actual

Aplicație funcțională cu pagini academice interactive și analiză transversală estimată în R pentru anul 2023.
