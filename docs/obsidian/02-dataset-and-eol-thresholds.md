# Dataset And EOL Thresholds

## Dataset principal

Datasetul principal pentru modelare este:

`data_test/Cleaned Datasets/Datasets 5-56 cleaned/cleaned_dataset/`

Continut important:

- `metadata.csv` - sursa de adevar pentru mapping-ul dintre cicluri si fisiere;
- `data/*.csv` - seriile temporale pe cicluri;
- `extra_infos/*.txt` - README-uri de provenienta pentru grupurile NASA.

Raw `.mat` si arhivele NASA raman pentru provenienta si verificari, nu pentru implementarea principala. Alegerea datasetului curatat pastreaza proiectul concentrat pe predictie RUL, nu pe reconstruirea unui parser MATLAB.

## Praguri EOL gasite in documentatia NASA

Documentatia NASA nu foloseste un singur prag universal pentru toate grupurile. Exemple relevante:

- B0005, B0006, B0007, B0018: EOL la 30% pierdere de capacitate, de la 2.0 Ah la 1.4 Ah, adica aproximativ 70% SOH.
- B0033, B0034, B0036: experimente pana la 1.6 Ah, adica 20% pierdere de capacitate si aproximativ 80% SOH.
- B0038, B0039, B0040: experimente pana la 1.6 Ah, adica aproximativ 80% SOH.
- B0041-B0048 si B0053-B0056: experimente pana la 1.4 Ah, adica aproximativ 70% SOH.
- B0049-B0052: experimentul s-a oprit din cauza unei erori software, deci EOL nu este la fel de curat.

## Decizie pentru lucrare

Lucrarea trebuie sa separe clar doua utilizari:

- benchmark-ul clasic NASA B0005/B0006/B0007/B0018 foloseste pragul NASA de 1.4 Ah / 70% SOH;
- demo-ul SOH-derived RUL poate folosi 80% SOH ca prag standard de sanatate, explicat ca prag practic si comparabil cu literatura.

Aceasta separare evita concluzii fortate si arata ca datasetul este eterogen.

## Scenarii de modelare

- `all_eligible`: benchmark strict cu toate bateriile care au suficiente cicluri utilizabile.
- `clean_benchmark`: benchmark principal pentru lucrare, cu baterii mai coerente si curbe mai interpretabile.
- `nasa_classic_4`: B0005, B0006, B0007, B0018, folosit pentru comparatii cu repository-uri si articole.
