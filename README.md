# Prepoznavanje umetničkih pokreta pomoću dubokih CNN mreža

Klasifikacija slika umetničkih dela u jedan od **18 umetničkih pokreta** — od vizantijske
ikonografije do pop arta — pomoću konvolucionih neuronskih mreža (CNN), na skupu podataka
**Pandora 18K**.

Projekat iz predmeta **Mašinsko učenje**, Matematički fakultet, Univerzitet u Beogradu.

**Autori:** Anja Jovanović, Mateja Stojanović

## Opis projekta

Prepoznavanje umetničkog pokreta je zahtevnije od klasične klasifikacije objekata: klase
nisu definisane sadržajem slike, već stilom — potezom kista, paletom, kompozicijom,
stepenom apstrakcije. Uz to su granice između pokreta postepene (impresionizam i
postimpresionizam se preklapaju i istorijski i vizuelno), pa je i za čoveka deo primera
sporan.

Na istom, fiksiranom skupu podataka poredimo dva pristupa:

1. **CNN trenirana od nule** — VGG-lite baseline (četiri Conv-Conv-Pool bloka) i hibridna
   arhitektura koja na tanji VGG stem dodaje Inception-stil granu sa paralelnim
   1×1 / 3×3 / 5×5 konvolucijama.
2. **Transfer learning** — pretrenirana ImageNet mreža (InceptionV3) sa novom
   klasifikacionom glavom, pa fine-tuning poslednjeg konvolucionog bloka.

Poređenje pokazuje koliko se dobija prenošenjem znanja sa ImageNet-a u domen u kojem
oblik objekta nije glavni signal.

Slučajno pogađanje za 18 klasa iznosi **5.56%**, što je donja granica za tumačenje svih
rezultata.

## Skup podataka

**Pandora 18K** — 18 klasa umetničkih pokreta, organizovanih hijerarhijski po pokretu i
zatim po autoru:

```
Pandora_18k/01_Byzantin_Iconography/Andrei Rublev/annunciation-1405.jpg
Pandora_18k/09_Impressionism/...
Pandora_18k/18_PopArt/...
```

Klase, hronološki: vizantijska ikonografija, rana renesansa, severna renesansa, visoka
renesansa, barok, rokoko, romantizam, realizam, impresionizam, postimpresionizam,
ekspresionizam, simbolizam, fovizam, kubizam, surrealizam, apstraktna umetnost, naivna
umetnost, pop art.

### Osnovna analiza

Detaljna analiza je u `notebooks/01_eda.ipynb`, sa figurama u `reports/figures/`. Ključna
zapažanja:

- **Ukupno 18 038 slika** u 18 klasa.
- Skup je **umereno nebalansiran**: najveća klasa je postimpresionizam (1 276 slika),
  najmanja fovizam (719). Odnos najveće i najmanje klase je oko **1.8 : 1** — dovoljno
  ravnomerno da nije bilo potrebe za preuzorkovanjem, ali dovoljno neravnomerno da
  accuracy treba čitati zajedno sa metrikama po klasi.
- Dimenzije slika se **jako razlikuju** (`reports/figures/size_scatter.png`,
  `size_histograms.png`) — od malih skenova do velikih reprodukcija, sa različitim odnosom
  širine i visine. Zato se sve slike skaliraju na fiksnih **224×224**.
- Vizuelno preklapanje između srodnih pokreta (impresionizam ↔ postimpresionizam, rana ↔
  visoka renesansa) vidi se već na uzorku po klasi (`reports/figures/sample_grid.png`) i
  očekuje se kao glavni izvor greške.

### Podela na trening, validaciju i test

| Skup | Slika | Udeo |
|---|---|---|
| trening | 12 626 | 70% |
| validacija | 2 706 | 15% |
| test | 2 706 | 15% |
| **ukupno** | **18 038** | 100% |

Podela je **stratifikovana po klasi** (svih 18 pokreta zastupljeno je u sva tri skupa u
istom odnosu) i **fiksirana pseudoslučajnim semenom** `SEED = 42`. Rezultat je zapisan u
`data/splits/train.csv`, `val.csv` i `test.csv` i **više se ne menja** — svi modeli se
treniraju i porede na identičnoj podeli.

Test skup se koristi **isključivo za finalnu evaluaciju**. Izbor arhitekture,
hiperparametara i broja epoha rađen je samo na osnovu validacionog skupa. Augmentacija
(horizontalno preslikavanje, rotacija do 10°, zoom do 10%) primenjuje se **samo na
trening**; validacija i test se samo skaliraju, da evaluacija bude stabilna i ponovljiva.

## Podešavanje okruženja

Potreban je Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Spisak paketa (`requirements.txt`): `tensorflow>=2.15`, `tf-keras`, `numpy<2`, `pandas`,
`matplotlib`, `scikit-learn`, `Pillow`, `jupyter`.

### Keras 2 režim (obavezno)

Na Pythonu 3.12 `pip` povlači TensorFlow 2.16 ili noviji, koji podrazumevano koristi **Keras 3**,
a Keras 3 je izbacio `ImageDataGenerator`, na kome stoji ceo `src/dataset.py`. Zato je uz
TensorFlow potreban paket `tf-keras` i promenljiva koja vraća stari API:

```bash
export TF_USE_LEGACY_KERAS=1
```

Najlakše je dopisati je u `.venv/bin/activate`, da važi pri svakoj aktivaciji. Provera:

```bash
python -c "import tensorflow as tf; print(tf.keras.__name__)"
```

Treba da ispiše `tf_keras...`. Ako piše `keras...`, promenljiva nije aktivna.

### GPU na WSL2

`pip install "tensorflow[and-cuda]"` instalira CUDA biblioteke, ali ih TensorFlow ne pronalazi
sam jer nisu na `LD_LIBRARY_PATH`. Rešenje je isto — dopisati u `.venv/bin/activate`:

```bash
export LD_LIBRARY_PATH="$(ls -d "$VIRTUAL_ENV"/lib/python*/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH"
```

**Skup podataka nije u repozitorijumu** (velik je, a deo slika je pod autorskim pravima —
vidi `Pandora_18k/Readme_Pandora18k.txt`). Folder `Pandora_18k/` treba raspakovati u koren
projekta, tako da putanje iz `data/splits/*.csv` budu ispravne. U git-u su samo liste
putanja, kod, metrike i figure.

Provera da okruženje i pipeline rade — ispisuje broj klasa i dimenzije jednog batch-a:

```bash
python src/dataset.py
```

Ako postoji GPU, dobro je proveriti da ga TensorFlow vidi:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Struktura projekta

```
src/
  dataset.py          ucitavanje fiksiranih split-ova + Keras generatori
  models.py           build_vgglite() i build_hybrid()
  transfer.py         InceptionV3: kesiranje feature-a, dense glava, fine-tuning
  train.py            trening skripta sa CLI argumentima
  evaluate.py         evaluacija sacuvanog modela na test skupu
  plot_curves.py      crtanje accuracy/loss krivih iz history.csv
notebooks/            sveske u redosledu pregledanja (01, 02, ...)
data/splits/          fiksirane train/val/test liste putanja
experiments/          po modelu: history.csv/json, run_config.json, test_metrics.json
reports/figures/      figure za izvestaj
```

Sveske se pregledaju po prefiksu:

| Sveska | Sadržaj |
|---|---|
| `01_eda.ipynb` | analiza skupa: broj slika po klasi, dimenzije, uzorci |
| `02_split.ipynb` | stratifikovana podela na trening/validaciju/test |
| `03_vgglite.ipynb` | VGG-lite arhitektura |
| `04_train_vgglite.ipynb` | treniranje VGG-lite modela |
| `05_vgglite_results.ipynb` | krive učenja i evaluacija VGG-lite modela |
| `06_transfer.ipynb` | transfer learning sa InceptionV3 |

## Pokretanje

Pregled arhitektura i broja parametara:

```bash
python src/models.py
```

Treniranje:

```bash
python src/train.py --model vgglite --epochs 20 --batch-size 64 --lr 1e-3
python src/train.py --model hybrid  --epochs 20
```

Argumenti: `--model {vgglite,hybrid}`, `--epochs`, `--batch-size`, `--lr`,
`--image-size`, `--steps`, `--val-steps`. Zadnja dva ograničavaju broj koraka po epohi i
korisni su za brzu proveru da sve radi, bez čekanja punog treninga.

Rezultati se upisuju u `experiments/custom_<model>/`:

- `best.keras` — težine sa najboljim `val_accuracy`
- `last.keras` — model posle poslednje epohe
- `history.csv`, `history.json` — metrike po epohi (trening i validacija)
- `run_config.json` — hiperparametri tog pokretanja

Krive učenja:

```bash
python src/plot_curves.py
```

### Transfer learning

InceptionV3 sa ImageNet težinama, zamrznutom bazom i novom glavom. Ulaz je **299×299**
(nativna veličina te mreže), za razliku od 224×224 kod custom modela.

Baza se ne prolazi iznova svake epohe — njeni 2048-dimenzioni izlazi se **keširaju** jednom
na disk, pa se glava trenira nad vektorima:

```bash
python src/transfer.py --stage cache          # jednokratno, sporo
python src/transfer.py --stage dense --epochs 100
python src/plot_curves.py --model transfer_dense
```

Bez argumenata (`python src/transfer.py`) rade se obe faze redom. Keš stoji u
`experiments/transfer_dense/features/` i nije u git-u.

Cena keširanja je što **augmentacija otpada** — svaka slika prolazi kroz bazu tačno jednom, pa
se protiv preprilagođavanja radi dropout-om u glavi (`--dropout`, podrazumevano 0.5).

Predobrada za InceptionV3 (opseg `[-1, 1]`) je ugrađena **u sam model**, kao prvi `Rescaling`
sloj, dok generatori i dalje daju `[0, 1]` kao za custom modele. Zato sačuvani model prima
sirove slike i evaluira se istom skriptom kao i ostali.

### Fine-tuning

Druga faza odmrzava **poslednji Inception blok** (od sloja `mixed9`) i uči ga zajedno sa
glavom, sa learning rate-om nižim za dva reda veličine:

```bash
python src/transfer.py --stage finetune
python src/plot_curves.py --model transfer_finetune
```

Rezultati idu u `experiments/transfer_finetune/`, odvojeno od dense faze. Glava se ne uči
iznova — inicijalizuje se iz `experiments/transfer_dense/head_best.keras`, jer bi nasumična
glava prvim gradijentima pokvarila pretrenirane težine baze.

Čim se baza odmrzne, **keš više ne važi** (izlazi baze se menjaju svake epohe), pa se ovde
ide kroz same slike. Epoha je zato bitno sporija nego u dense fazi, ali augmentacija sada
ponovo radi i koristi se ista `dataset.make_generators` konvencija kao kod custom modela.

`BatchNormalization` slojevi ostaju zamrznuti i baza se poziva sa `training=False` — sa malim
batch-om bi svežih nekoliko epoha pokvarilo ImageNet statistike i model bi propao.

Podrazumevano: `--ft-lr 1e-5`, `--ft-epochs 30`, `--ft-batch-size 16`, early stopping po
`val_accuracy` sa `--patience 6`. Ako pukne memorija GPU-a, spustiti `--ft-batch-size`.

Evaluacija na test skupu — pokreće se **tek na kraju**, nad sačuvanim `best.keras`:

```bash
python src/evaluate.py --model vgglite
python src/evaluate.py --model hybrid
python src/evaluate.py --model transfer_finetune --exp-dir experiments/transfer_finetune
```

Veličina slike se čita iz `run_config.json` tog eksperimenta, da bi se poklopila sa
treningom. Skripta dopisuje u `experiments/custom_<model>/`:

- `test_metrics.json` — test accuracy/loss, macro i weighted F1, broj parametara
- `classification_report.txt`, `.json` — precision/recall/F1 po klasi
- `confusion_matrix.csv` i `confusion_matrix.png` (kopija figure ide i u
  `reports/figures/<model>_confusion.png`)

## Literatura

- C. Florea, C. Toca, F. Gieske. *Artistic Movement Recognition by Boosted Fusion of Color
  Structure and Topographic Description.* IEEE Winter Conference on Applications of
  Computer Vision (WACV), Santa Rosa, CA, USA, 2017. — rad koji opisuje Pandora 18K skup i
  koji autori skupa traže da se citira.
- Y. Yu, O. Jin, D. Hsu. *Artistic Movement Recognition using Deep CNNs.* Stanford
  University, CS231n, 2017.
  [cs231n.stanford.edu/reports/2017/pdfs/411.pdf](https://cs231n.stanford.edu/reports/2017/pdfs/411.pdf)
  — polazna tačka za temu i arhitekture; njihova hibridna VGG + Inception mreža trenirana
  od nule postiže 31.2%, a najbolji rezultat transfer learning-om na InceptionV3 iznosi
  56.6%.
- Dokumentacija Keras / TensorFlow: `ImageDataGenerator`, `InceptionV3`, transfer learning
  i fine-tuning.
