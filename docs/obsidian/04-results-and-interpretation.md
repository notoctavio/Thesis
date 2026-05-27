# Results And Interpretation

## Direct RUL

Rezultatele curente arata ca modelele tree-based sunt cele mai puternice pentru predictia directa RUL pe feature-urile extrase.

Rezultate importante dupa corectiile v2:

- `clean_benchmark`: ExtraTrees a obtinut aproximativ R2 = 0.814 si RMSE = 22.26 cicluri.
- `clean_benchmark`: LSTM este competitiv, cu aproximativ R2 = 0.788 si RMSE = 23.75 cicluri.
- `all_eligible`: CNN-LSTM s-a imbunatatit dupa curatarea v2, dar ExtraTrees ramane mai bun.
- `all_eligible`: rezultatele sunt mai slabe, deoarece datasetul este eterogen si include baterii cu comportament mai dificil.
- `nasa_classic_4`: modelele clasice obtin scoruri foarte mari, ceea ce explica de ce multe repository-uri publice raporteaza rezultate aproape perfecte pe subsetul clasic NASA.

## SOH / capacity prediction

SOH prediction este cea mai puternica directie practica:

- `clean_benchmark`: RandomForest a obtinut aproximativ R2 = 0.982 si RMSE = 0.0098 SOH.
- `nasa_classic_4`: RandomForest a obtinut aproximativ R2 = 0.994 si RMSE = 0.0064 SOH.
- Dupa reformularea reziduala a modelelor secventiale SOH, CNN-LSTM si LSTM au devenit competitive pe `clean_benchmark`, cu R2 aproximativ 0.981 si RMSE aproximativ 0.010 SOH.
- Pe `all_eligible`, modelele secventiale ajung aproape de modelele clasice, dar datasetul ramane mai eterogen si mai dificil.
- Pe `nasa_classic_4`, LSTM se imbunatateste clar, dar CNN-LSTM ramane instabil pe split-ul foarte mic; acesta este raportat ca rezultat comparativ, nu ca model recomandat.

Aceste rezultate sunt bune, dar trebuie interpretate corect. SOH este o curba mai neteda decat RUL direct, iar o predictie foarte buna a SOH nu inseamna automat ca viitorul complet al bateriei este prezis perfect.

## Concluzie metodologica

Pentru lucrare, prezentarea cea mai corecta este:

1. direct RUL prediction ca experiment principal comparativ;
2. SOH prediction ca formulare practica si stabila;
3. derived RUL ca rezultat aplicabil in demo.

Nu trebuie sa sustinem ca LSTM este automat mai bun. Rezultatele curente arata ca modelele clasice, bine alimentate cu feature-uri, sunt foarte robuste, iar modelele secventiale devin competitive cand problema este formulata ca predictie a degradarii incrementale a SOH.
