# RUL And SOH Debugging Notes

## Probleme gasite

1. Unele predicții RUL aratau foarte rau deoarece app-ul putea selecta implicit primul model alfabetic, nu modelul cel mai bun dupa RMSE.
2. Feature table-ul initial folosea `battery_cycle_features_v1.csv`; unele grupuri NASA aveau un prim ciclu de capacitate anormal de mic sau mare. Asta facea ca SOH/EOL sa para trecut la ciclul 1 pentru baterii care de fapt aveau o curba utila dupa startup.
3. Notebook-ul SOH folosea `bfill()` pentru primul `prev_soh`, ceea ce introducea informatie din viitor pentru primul rand al fiecarei baterii.
4. RUL direct este un benchmark de cicluri ramase pana la finalul curbei inregistrate. Pentru output practic si aliniat fizic, demo-ul trebuie sa puna in fata SOH-derived RUL.
5. SOH nu avea inca LSTM/CNN-LSTM; comparatia era dezechilibrata fata de RUL.
6. Prima varianta LSTM/CNN-LSTM pentru SOH prezicea direct SOH absolut, iar CNN-LSTM generaliza slab pe `clean_benchmark`.

## Reparatii facute

- S-a introdus feature table v2: `artifacts/features/battery_cycle_features_v2.csv`.
- Primul ciclu de capacitate este tratat ca startup anomaly daca este sub 85% sau peste 115% fata de platoul imediat urmator.
- Initial capacity se calculeaza din primele 5 cicluri curatate, nu doar din primele 3.
- SOH lag features folosesc doar informatie din trecut; primul ciclu primeste fallback nominal `prev_soh = 1.0`.
- Modelele tree-based din notebook-uri folosesc `n_jobs=1` pentru reproductibilitate locala.
- Streamlit selecteaza implicit modelul recomandat dupa RMSE, nu primul alfabetic.
- S-a adaugat `scripts/train_sequence_soh.py` si `notebooks/05_sequence_soh_prediction.ipynb` pentru LSTM/CNN-LSTM pe SOH.
- Modelele secventiale SOH au fost reformulate sa prezica `soh_delta_target = soh - prev_soh`, apoi SOH-ul este reconstruit ca `prev_soh + pred_delta_soh`. Metricile raman calculate pe SOH reconstruit.

## Rezultate dupa reparatii

Direct RUL:

- `all_eligible`: ExtraTrees ramane cel mai bun, RMSE aproximativ 18.48 cicluri, R2 aproximativ 0.593.
- `clean_benchmark`: ExtraTrees ramane cel mai bun, RMSE aproximativ 22.26 cicluri, R2 aproximativ 0.814.
- `nasa_classic_4`: HistGradientBoosting ramane foarte bun, RMSE aproximativ 3.05 cicluri, R2 aproximativ 0.996.

SOH:

- `clean_benchmark`: RandomForest ramane cel mai bun, RMSE aproximativ 0.0098 SOH, R2 aproximativ 0.982.
- `nasa_classic_4`: RandomForest ramane cel mai bun, RMSE aproximativ 0.0064 SOH, R2 aproximativ 0.994.
- Dupa reformularea reziduala, `clean_benchmark` CNN-LSTM ajunge la RMSE aproximativ 0.0100 SOH si R2 aproximativ 0.981, iar LSTM este foarte apropiat.
- Pe `all_eligible`, LSTM/CNN-LSTM ajung in jur de R2 = 0.77, foarte aproape de modelele clasice.
- Pe `nasa_classic_4`, LSTM ajunge la R2 aproximativ 0.946, dar CNN-LSTM ramane instabil si poate avea R2 negativ pe test.

## Interpretare

Nu este o greseala ca unele modele LSTM/CNN-LSTM nu sunt cele mai bune. Datasetul are putine baterii, protocoale diferite si split pe baterii nevazute. Modelele tree-based folosesc feature-uri agregate foarte informative si generalizeaza stabil. Reformularea reziduala face modelele secventiale mult mai competitive pe SOH, dar nu trebuie prezentata ca predictie industriala long-horizon complet autonoma.

Pentru lucrare, concluzia corecta este:

- direct RUL ramane experiment comparativ;
- SOH-derived RUL ramane outputul practic principal;
- deep learning este investigat si raportat onest, nu fortat ca model castigator.
