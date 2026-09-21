"""Articolele de literatură definite direct în backend."""

from __future__ import annotations

from typing import Any


ARTICOLE: list[dict[str, Any]] = [
    {
        "id": "koyuncu_oksak_2021",
        "titlu": "Does More Inflation Mean More Female Labor Force Participation? The Case of Turkey",
        "autori": "Cüneyt Koyuncu și Yüksel Okşak",
        "an": 2021,
        "tara": "Turcia",
        "perioada": "1990–2019",
        "frecventa": "Anuală",
        "sursa_date": "World Development Indicators – World Bank",
        "metoda": "ARDL – Autoregressive Distributed Lag",
        "model": "ARDL(3,4)",
        "categorie": "Inflație și factori macroeconomici",
        "status": "Citit",
        "variabile_dependente": [
            "rata participării femeilor cu vârsta de 15 ani și peste",
            "rata participării femeilor cu vârsta între 15 și 64 de ani",
        ],
        "variabile_explicative": [
            "inflația, măsurată prin deflatorul PIB",
        ],
        "idei_principale": [
            {
                "tema": "Inflație și puterea de cumpărare",
                "accent": "plum",
                "text": "Inflația poate reduce puterea de cumpărare a gospodăriilor și poate determina femeile să intre pe piața muncii pentru a suplimenta venitul familiei.",
            },
            {
                "tema": "Relație pozitivă pe termen lung",
                "accent": "teal",
                "text": "Autorii identifică o relație pozitivă între inflație și participarea femeilor pe termen lung, pentru ambele definiții ale participării feminine.",
            },
            {
                "tema": "Termen scurt",
                "accent": "gold",
                "text": "Modificările inflației nu sunt semnificative pe termen scurt, astfel încât rezultatul principal al articolului privește relația estimată pe termen lung.",
            },
        ],
        "teste": [
            "testul Augmented Dickey–Fuller – ADF",
            "selecția lagurilor pe baza criteriului AIC",
            "ARDL Bounds Test pentru cointegrare",
            "modelul Error Correction – ECM",
            "testul Breusch–Godfrey pentru autocorelare",
            "testul Breusch–Pagan–Godfrey pentru heteroscedasticitate",
            "testul Ramsey RESET pentru specificarea modelului",
            "testul CUSUM pentru stabilitatea parametrilor",
        ],
        "rezultate": [
            "coeficientul inflației pentru femeile de 15+ ani: aproximativ 0,109",
            "coeficientul inflației pentru femeile de 15–64 ani: aproximativ 0,138",
            "ambii coeficienți sunt pozitivi și semnificativi statistic pe termen lung",
            "termenii ECM indică ajustarea anuală a aproximativ unei treimi din abaterea față de echilibrul estimat",
            "inflația nu este semnificativă statistic pe termen scurt",
        ],
        "coeficienti": [
            {"grup": "Femei 15+", "valoare": "0,109"},
            {"grup": "Femei 15–64", "valoare": "0,138"},
        ],
        "observatii_critice": [
            "inflația este clasificată de autori ca fiind integrată de ordinul al doilea, I(2), ceea ce ridică probleme pentru aplicarea formei standard a ARDL Bounds Test",
            "modelul conține o singură variabilă explicativă și poate omite determinanți importanți precum PIB-ul, șomajul, educația, fertilitatea și urbanizarea",
            "eșantionul conține numai aproximativ 30 de observații, în timp ce modelul ARDL(3,4) utilizează mai multe laguri",
            "rezultatele indică o asociere pe termen lung, nu demonstrează automat o relație cauzală",
            "CPI ar putea fi o măsură mai apropiată de puterea de cumpărare a gospodăriilor decât deflatorul PIB",
        ],
        "utilitate_proiect": [
            "justifică păstrarea inflației în setul inițial de variabile",
            "oferă un mecanism economic bazat pe reducerea puterii de cumpărare",
            "arată necesitatea separării efectelor pe termen scurt de cele pe termen lung",
            "justifică folosirea mai multor variabile de control",
            "oferă un punct de comparație pentru analiza economiilor europene în perioada 2001–2023",
        ],
    }
]

