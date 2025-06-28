# Detekcja budynków na obrazach z zastosowaniem YOLO: implementacja i porównanie modeli

## 1. Cel projektu
Celem niniejszego projektu jest opracowanie narzędzia umożliwiającego automatyczne wykrywanie budynków na obrazach, z jednoczesnym zaznaczaniem ich położenia za pomocą prostokątnych obszarów zainteresowania (ROI – Region of Interest). Integralną częścią projektu jest również przeprowadzenie analizy porównawczej wybranych wariantów architektury YOLO (You Only Look Once) w kontekście skuteczności i dokładności detekcji obiektów.

## 2. Środowisko i sprzęt
Plik `environment.yml` definiuje środowisko Anaconda o nazwie `MLR_env`, zawierające niezbędne biblioteki do pracy z danymi i trenowania modeli opartych na PyTorch.
```
name: MLR_env
channels:
  - nvidia
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - numpy
  - pandas
  - scipy
  - opencv
  - pillow
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=12.1
  - pip
  - pip:
    - ultralytics
```
Wśród nich są m.in. NumPy, Pandas, SciPy, OpenCV, Pillow oraz rozszerzenia PyTorch: torchvision i torchaudio. Środowisko wspiera akcelerację GPU dzięki pytorch-cuda w wersji 12.1. Dodatkowo zainstalowano bibliotekę Ultralytics, umożliwiającą pracę z modelami YOLO.

Trenowanie modeli odbywało się na komputerze z kartą NVIDIA RTX 2060 i procesorem AMD Ryzen 5 3600, co pozwoliło na efektywne wykorzystanie CUDA i mixed precision, choć w przypadku największego modelu YOLOv11x konieczne było zmniejszenie batch size ze względu na ograniczenia pamięci GPU.

## 3. Dataset
W projekcie wykorzystano zestaw danych będący połączeniem trzech źródeł zawierających zdjęcia z oznaczonymi budynkami. Pierwszy z nich to ogólnodostępny zbiór danych: Building detection – Roboflow (https://universe.roboflow.com/trees-detection/building-detection-scazo), zawierający obrazy z wcześniej zaznaczonymi obszarami ROI (Region of Interest) wokół budynków.

Drugim źródłem jest zestaw Modern Architecture – Kaggle (https://www.kaggle.com/datasets/tompaulat/modernarchitecture). Zbiór ten zawiera obrazy nowoczesnych obiektów architektonicznych, które zostały szczegółowo opisane w nazwach plików. Na podstawie tych opisów przeprowadzono filtrację zdjęć, pozostawiając jedynie te przedstawiające zewnętrzne elementy budynków. Następnie, przy użyciu narzędzia Roboflow, ręcznie oznaczono widoczne budynki poprzez wyznaczenie obszarów ROI.

Trzecim uzupełniającym źródłem danych był zbiór House, Rooms, Streets Image Dataset – Kaggle (https://www.kaggle.com/datasets/mikhailma/house-rooms-streets-image-dataset). Ze zbioru tego wyselekcjonowano zdjęcia przedstawiające ulice, które również zawierały widoczne budynki. Na wybranych obrazach ręcznie zaznaczono obszary budynków, uzupełniając tym samym zbiór treningowy o różnorodne sceny miejskie.

Nowo stworzony dataset dostępny jest na platformie Roboflow (https://universe.roboflow.com/buildingrecognition/building-detection-id8vh).

Po przygotowaniu adnotacji, wszystkie trzy zbiory zostały połączone w jeden, liczący łącznie ponad 3000 obrazów (po zastosowaniu augumentacji). Dane te zostały następnie losowo podzielone na zbiór treningowy (85%) oraz walidacyjny (15%) przy użyciu skryptu `split_maker.py`. Dodatkowo, ze zbioru treningowego losowo wybrano 100 zdjęć, tworząc niezależny zestaw testowy, który posłużył do końcowej oceny wydajności modeli detekcji.

## 4. Preprocessing i augumentacja
Do wstępnego przetwarzania oraz augmentacji obrazów zastosowano zestaw parametrów mających na celu zwiększenie różnorodności danych treningowych i poprawę ogólności modelu. Proces ten został zastosowany wyłącznie do danych pochodzących z ręcznie przygotowanych zbiorów (tj. na podstawie datasetów z Kaggle), natomiast gotowy zbiór danych z platformy Roboflow (Building Detection – Roboflow) nie był poddawany żadnej dodatkowej obróbce.

Wśród zastosowanych operacji preprocessingowych znalazło się automatyczne wyrównanie orientacji obrazów (Auto-Orient) oraz skalowanie wszystkich zdjęć do wymiarów 640x640 pikseli poprzez rozciągnięcie (Resize: Stretch). Każdy przykład treningowy generował trzy różne warianty danych wyjściowych (Outputs per training example: 3), co skutecznie zwiększyło objętość i różnorodność danych.

W zakresie augmentacji przestrzennej wprowadzono losową rotację obrazów w zakresie od -15° do +15° oraz deformacje typu shear – maksymalnie ±10° w poziomie i w pionie. Dodatkowo, do 19% obrazów zastosowano konwersję do skali szarości (Grayscale), co pozwalało modelowi lepiej radzić sobie z brakiem informacji kolorystycznej. Wprowadzono również subtelne zakłócenia wizualne: rozmycie (Blur) do 2.6 piksela oraz szum (Noise) na poziomie do 0.46% pikseli. Zabiegi te miały na celu zwiększenie odporności modelu na niedoskonałości danych wejściowych.

Taki zestaw operacji preprocessingowych i augmentacyjnych pozwolił na znaczące rozszerzenie i urozmaicenie zbioru treningowego, co jest szczególnie istotne w przypadku danych tworzonych ręcznie i o mniejszej skali niż gotowe zbiory benchmarkowe.

## 5. Modele
W ramach projektu przeprowadzono porównanie pięciu modeli detekcji obiektów z rodziny YOLO, należących do różnych generacji i reprezentujących różne rozmiary architektury. W eksperymentach uwzględniono modele: YOLOv11n, YOLOv11m, YOLOv11x, YOLOv8n oraz YOLOv5n.

Modele z serii YOLOv11 reprezentują najnowszą generację algorytmu, oferującą ulepszenia w zakresie architektury i dokładności detekcji. Wersje n, m i x różnią się liczbą parametrów oraz złożonością – od lekkiego i szybkiego YOLOv11n, przez średniej wielkości YOLOv11m, po najbardziej zaawansowany model YOLOv11x, oferujący najwyższą dokładność kosztem większych zasobów obliczeniowych.

Dla porównania, do testów włączono również dwa modele z wcześniejszych generacji: YOLOv8n oraz YOLOv5n. Oba należą do wersji „nano” – lekkich wariantów swoich serii, zoptymalizowanych pod kątem szybkości działania i mniejszych urządzeń.

Celem porównania było określenie, który z modeli najlepiej sprawdza się w zadaniu detekcji budynków na przygotowanym zbiorze danych, z uwzględnieniem zarówno dokładności (np. mAP), jak i efektywności obliczeniowej (czas inferencji, wykorzystanie GPU).

## 6. Uczenie
Plik `learn.py` odpowiada za proces trenowania pięciu modeli detekcji obiektów z rodziny YOLOv11, YOLOv8 oraz YOLOv5 przy użyciu biblioteki Ultralytics. Skrypt został zaprojektowany tak, aby automatycznie inicjalizować i trenować pięć wariantów modelu: YOLOv11n, YOLOv11m, YOLOv11x, YOLOv8n oraz YOLOv5n, które różnią się skalą, złożonością oraz wiekiem architektury.

Fragment pliku `learn.py` pokazujący inicjalizację modeli oraz zastosowane parametry dla modelu YOLOv11n:

```python
# Load YOLO models with different pre-trained weights
model_11n = YOLO('yolo11n.pt')
model_11m = YOLO('yolo11m.pt')
model_11x = YOLO('yolo11x.pt')
model_8n = YOLO('yolov8n.pt')
model_5n = YOLO('yolov5n.pt')

# Train yolo11n model
model_11n.train(
    data='datasets/data.yaml',
    epochs=50,            # Number of training epochs
    imgsz=640,            # Input image size (pixels)
    amp=True,             # Use mixed precision for faster training
    patience=10,          # Early stopping patience (epochs)
    lr0=0.01,             # Initial learning rate
    batch=32,             # Batch size (number of images per batch)
    workers=4             # Number of data loader worker threads
)
```
Każdy z modeli trenowany jest na tym samym zbiorze danych opisanym w pliku data.yaml, ale z różnymi hiperparametrami dostosowanymi do ich wielkości. Parametry te obejmują:
- liczbę epok (epochs) – im większy model, tym dłużej trwa trening (50 epok dla 11n, 120 dla 11x),
- początkową wartość współczynnika uczenia (lr0) – dostosowaną indywidualnie do każdego modelu,
- rozmiar wsadu (batch) – mniejszy dla bardziej złożonych modeli, aby dopasować się do ograniczeń pamięci GPU,
- mixed precision training (amp=True) – przyspiesza trening przy mniejszym zużyciu pamięci,
- early stopping (patience=10) – zatrzymuje trening, jeśli model nie poprawia się przez 10 epok.

## 7. Predykcje
<img src="runs\detect\predict6\combined.png" width="800"/>

*Wyniki części predykcji YOLOv11n*


<img src="runs\detect\predict7\combined.png" width="800"/>

*Wyniki części predykcji YOLOv11m*


<img src="runs\detect\predict8\combined.png" width="800"/>

*Wyniki części predykcji YOLOv11x*


<img src="runs\detect\predict9\combined.png" width="800"/>

*Wyniki części predykcji YOLOv8n*


<img src="runs\detect\predict10\combined.png" width="800"/>

*Wyniki części predykcji YOLOv5n*

## 8. Porównanie
W przeprowadzonym porównaniu pięciu modeli YOLO (11n, 11m, 11x, 8n oraz 5n) zestawiono dokładności detekcji (mAP) na różnych progach oraz czasu przetwarzania poszczególnych etapów inferencji.

Wyniki przedstawiono w poniższej tabeli:
| Model | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 | Preprocess [ms] | Inference [ms] | Postprocess [ms] |
|-------|--------------|---------|----------|-----------------|----------------|------------------|
| 11n   | 0.6596       | 0.9038  | 0.7422   | 2.0             | 8.0            | 2.7              |
| 11m   | 0.6055       | 0.8601  | 0.6770   | 1.9             | 19.1           | 1.7              |
| 11x   | 0.6033       | 0.8437  | 0.6869   | 1.8             | 45.7           | 1.7              |
| 8n    | 0.6709       | 0.9116  | 0.7821   | 1.9             | 6.0            | 1.7              |
| 5n    | 0.6428       | 0.8922  | 0.7403   | 1.9             | 6.5            | 1.8              |


Na podstawie wartości mAP można zauważyć, że model 8n osiąga najwyższą dokładność, zarówno w szerokim zakresie progów (mAP@0.5:0.95 = 0.6709), jak i na poziomie mAP@0.5 oraz mAP@0.75, co świadczy o jego lepszej zdolności do precyzyjnego wykrywania budynków na zdjęciach.

Modele 11n oraz 5n również wykazują dobre wyniki, plasując się tuż za modelem 8n. W szczególności 11n cechuje się wysoką wartością mAP@0.5 (0.9038), co oznacza, że jest skuteczny w wykrywaniu budynków z minimalnymi wymaganiami co do dokładności lokalizacji. Z kolei modele 11m i 11x osiągają niższe wyniki pod względem mAP, zwłaszcza w zakresie mAP@0.5:0.95, co może wskazywać na konieczność dalszej optymalizacji lub lepszego dostosowania parametrów treningowych.

Pod względem szybkości działania najszybsze modele pod względem czasu inferencji to 8n i 5n, które potrzebują około 6 ms na etap inferencji, co jest istotne przy zastosowaniach wymagających niskich opóźnień. Modele z rodziny 11, szczególnie 11x, mają zauważalnie dłuższy czas inferencji (do 45,7 ms), co może być efektem ich większej złożoności i rozmiaru. Warto zauważyć, że czas przetwarzania na etapach preprocess i postprocess jest porównywalny dla wszystkich modeli i stanowi mniejszą część całkowitego czasu inferencji.

Podsumowując, wybór modelu zależy od kompromisu między dokładnością a szybkością działania. Model 8n wydaje się oferować najlepszy balans, łącząc wysoką skuteczność detekcji z niskim czasem inferencji, podczas gdy modele 11m i 11x wymagają dalszej optymalizacji, aby osiągnąć konkurencyjną wydajność.

## 9. Wnioski
W przeprowadzonym projekcie udało się skutecznie zastosować modele z rodziny YOLO do automatycznego wykrywania budynków na obrazach. Proces trenowania oraz testowania modeli potwierdził ich zdolność do precyzyjnego zaznaczania obiektów w postaci prostokątnych obszarów zainteresowania (ROI). Analiza wyników pokazała, że w tym konkretnym zadaniu lepsze rezultaty osiągnęły mniejsze warianty modeli YOLO, które nie tylko zapewniły wyższą dokładność detekcji, ale również charakteryzowały się krótszym czasem inferencji.

Jedną z głównych przyczyn może być niedostateczne dopasowanie hiperparametrów treningowych – większe modele wymagają zwykle niższego learning rate oraz dłuższego czasu uczenia, aby w pełni wykorzystać swój potencjał. Przykładowo, YOLOv11x był trenowany przez 120 epok przy learning rate 0.001, co może być niewystarczające.

Drugim czynnikiem mogącym wpływać na wyniki jest rozmiar zbioru danych. Modele takie jak YOLOv11x mają więcej parametrów i wymagają znacznie większej ilości zróżnicowanych danych treningowych, aby uniknąć przeuczenia lub niedouczenia. W projekcie użyto około 3 tysięcy zdjęć, co może być wystarczające dla mniejszych modeli, ale zbyt małe dla większych architektur. W przypadku ograniczonych danych większe modele mogą po prostu nie nauczyć się wystarczająco dobrze rozpoznawać istotne wzorce.

Kolejnym aspektem, który mógł osłabić jakość uczenia modeli YOLOv11m i YOLOv11x, jest zastosowanie niskiego batch size. Model YOLOv11x trenowano z batch size równym 4, co przy dużych modelach może prowadzić do niestabilności podczas treningu. Również augmentacja danych może mieć wpływ – większe modele są bardziej wrażliwe na jakość i różnorodność danych wejściowych, dlatego nieoptymalna augmentacja może utrudniać generalizację.

Aby poprawić wyniki YOLOv11m i YOLOv11x, warto przede wszystkim zwiększyć liczbę danych treningowych, zarówno poprzez dodanie nowych przykładów, jak i przez zastosowanie bardziej zaawansowanych metod augmentacji (np. CutMix, Mosaic). Kolejnym krokiem może być wydłużenie treningu i obniżenie wartości lr0 (np. do 0.0005 lub niżej). Wskazane jest również zwiększenie batch size, ale niestety jest to niemożliwe przez ograniczenia sprzętowe (6 GB VRAM). Dobrym rozwiązaniem byłoby także wstępne wytrenowanie modeli na większym, ogólnym zbiorze (np. COCO), a dopiero potem ich dostrojenie na docelowym zbiorze – czyli zastosowanie transfer learningu.
