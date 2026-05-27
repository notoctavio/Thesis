# Experiment Design

## Intrebarea principala

Cat de bine poate un model de invatare automata sa estimeze durata de viata utila ramasa a unei baterii Li-Ion, folosind istoricul ciclurilor de incarcare/descarcare si indicatorii de degradare?

## Tinte de predictie

### Direct RUL

Tinta este `rul_cycles`, adica numarul de cicluri ramase pana la sfarsitul secventei utile. Aceasta formulare este intuitiva, dar poate fi dificila pe dataseturi eterogene.

### SOH / capacity prediction

Tinta este `soh` sau raportul dintre capacitatea curenta si capacitatea initiala. Dupa predictia SOH, RUL este derivat prin estimarea momentului in care curba ajunge sub pragul EOL.

Aceasta formulare este mai stabila deoarece capacitatea se degradeaza relativ gradual.

Pentru modelele secventiale SOH, formularea imbunatatita foloseste o tinta reziduala:

```text
soh_delta_target = soh - prev_soh
pred_soh = prev_soh + pred_delta_soh
```

Aceasta pastreaza evaluarea pe SOH, dar ajuta LSTM/CNN-LSTM sa invete rata de degradare dintre cicluri, nu valoarea absoluta a SOH de la zero.

## Split de evaluare

Evaluarea principala foloseste battery-level holdout:

- unele baterii sunt folosite la antrenare;
- alte baterii sunt folosite la validare;
- alte baterii sunt tinute pentru test.

Aceasta strategie este mai corecta decat random split pe randuri, deoarece evita ca acelasi comportament al aceleiasi baterii sa fie prezent si in train, si in test.

## Modele comparate

Modele clasice:

- SVR cu kernel RBF;
- Random Forest;
- Extra Trees;
- HistGradientBoosting;
- GradientBoosting pentru SOH.

Modele secventiale:

- LSTM;
- CNN-LSTM.

Transformer/attention ramane directie viitoare, nu nucleul lucrarii, deoarece ar adauga complexitate mare fara garantie ca imbunatateste rezultatele pe datasetul curent.

## Metrici

Metricile raportate:

- RMSE;
- MAE;
- R2.

Pentru RUL, RMSE si MAE sunt in cicluri. Pentru SOH, erorile sunt in unitati SOH.
