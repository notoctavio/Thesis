# Final Project Overview

## Ce construieste proiectul

Proiectul final este un workflow complet de data science pentru predictia RUL a bateriilor Li-Ion:

1. analiza datasetului NASA;
2. curatare si feature engineering;
3. definirea pragurilor SOH/EOL;
4. antrenarea si evaluarea modelelor;
5. comparatie intre modele clasice si secventiale;
6. demo Streamlit pentru explorarea predictiilor.

## Cum functioneaza demo-ul Streamlit

Demo-ul final este gandit ca o prezentare ghidata, nu ca un panou tehnic. Prima pagina trebuie sa arate direct povestea:

Date NASA -> predictie SOH -> prag EOL -> RUL derivat.

Aplicatia fixeaza explicit mesajul lucrarii: SOH-derived RUL este abordarea principala, iar RUL direct ramane benchmark comparativ pentru a arata de ce formularea bazata pe SOH este mai stabila si mai interpretabila.

Default-ul pentru prezentare este:

- scenariu: `nasa_classic_4`;
- task: SOH-derived RUL;
- baterie: `B0007`, cand este disponibila;
- prag demo: 80% SOH pentru a arata RUL in curba disponibila B0007;
- prag documentat NASA classic: 70% SOH, selectabil si explicat metodologic.

Aplicatia trebuie sa permita in continuare:

- selectarea scenariului: `clean_benchmark`, `all_eligible`, `nasa_classic_4`;
- selectarea taskului: direct RUL sau SOH-derived RUL;
- selectarea modelului disponibil;
- selectarea bateriei si a ciclului curent;
- afisarea curbei reale vs prezise;
- afisarea pragului EOL pentru SOH;
- afisarea metricilor modelului;
- afisarea unei interpretari scurte pentru scenariul selectat.

Controalele tehnice trebuie sa ramana in tabul de explorare avansata, ca primul ecran sa fie usor de inteles pentru comisie.

## Ce arata practic demo-ul

Pentru o baterie aleasa, utilizatorul poate simula intrebarea:

La ciclul curent X, care este starea bateriei si cate cicluri mai pot ramane pana la sfarsitul vietii utile?

Pentru bateriile din test, demo-ul poate compara estimarea cu valoarea reala, deoarece avem ground truth.

## MLOps minim

Proiectul nu are nevoie de MLOps cloud complet. Este suficient un MLOps local si reproductibil:

- modele salvate in `artifacts/models`;
- metrici salvate in `artifacts/metrics`;
- predictii salvate in `artifacts/predictions`;
- notebook-uri executabile;
- module `src/` pentru logica reutilizabila;
- demo care incarca artefactele salvate;
- teste rapide pentru validarea artefactelor si a helperelor.

MLOps complet cu registry, monitorizare drift si deployment cloud poate fi mentionat la directii viitoare.
