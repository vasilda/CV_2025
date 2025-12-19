# <b> Лабораторная работа №3 </b>

<b> Цель работы: </b>

Научиться создавать простые системы классификации изображений на основе сверточных нейронных сетей.


<b> Задание: </b>
1. Выбрать цель для задачи классификации и датасет (train/val: собрать либо найти, например, на Kaggle, test: собрать, разметить, не менее 50 изображений).
2. Зафиксировать архитектуру сети, loss, метрики качества.
3. Натренировать (либо дотренировать сеть) на выбранном датасете
4. Оценить качество работы по выбранной метрике на валидационной выборке, определить, не переобучилась ли модель.
5. Сделать отчёт в виде readme на GitHub, там же должен быть выложен исходный код.

## <b> Теоретическая база </b>
### <b> Задача бинарной классификации изображений </b>
Данный проект решает задачу бинарной классификации изображений на два класса: "кошки" и "собаки". 
Основу решения составляет сверточная нейронная сеть (Convolutional Neural Network), которая состоит из:
1. Сверточных слоев: автоматически извлекают иерархические признаки из изображений
2. Пулинговых слоев: уменьшают пространственные размерности, сохраняя важные признаки
3. Полносвязных слоев: выполняют окончательную классификацию на основе извлеченных признаков

<b> Архитектура модели </b>

Модель CatsDogsCNN состоит из следующих компонентов:
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│      <span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">200</span>, <span style="color: #00af00; text-decoration-color: #00af00">200</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">200</span>, <span style="color: #00af00; text-decoration-color: #00af00">200</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">896</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">200</span>, <span style="color: #00af00; text-decoration-color: #00af00">200</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │        <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">256</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │        <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">100</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)  │        <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │           <span style="color: #00af00; text-decoration-color: #00af00">512</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">147,584</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>                     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>, <span style="color: #00af00; text-decoration-color: #00af00">50</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)    │       <span style="color: #00af00; text-decoration-color: #00af00">295,168</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling2D</span>     │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │       <span style="color: #00af00; text-decoration-color: #00af00">131,584</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │         <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │       <span style="color: #00af00; text-decoration-color: #00af00">131,328</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │         <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│      <span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">257</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>

<b>

Total params: 2,515,973 (9.60 MB)

Trainable params: 838,017 (3.20 MB)

Non-trainable params: 1,920 (7.50 KB)

Optimizer params: 1,676,036 (6.39 MB)
</b>

## <b> Алгоритм работы системы </b>
<b> Подготовка данных </b>

Для разделение готового датасета (https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) на тренировочную и валидационную выборки используется функция split_dataset, которая:
1. Собирает все изображения из исходной директории;
2. Разделяет данные в соотношении  80%/20%;
3. Копирует файлы в соответствующие директории.

Тестовые данные собирались самостоятельно.

Затем используется ImageDataGenerator для:
1. Нормализации значений пикселей (деление на 255);
2. Генерации батчей данных;
3. Автоматической маркировки классов.

<b> Обучение модели </b>

Оптимизатор: Adam с learning_rate = 1e-4

Функция потерь: binary_crossentropy

Метрика: accuracy

Коллбэки для улучшения обучения:
1. ModelCheckpoint для сохранение лучшей модели;
2. EarlyStopping для остановка при отсутствии улучшений;
3. ReduceLROnPlateau для уменьшение learning rate;
4. CSVLogger для логирование процесса обучения.

## <b> Результаты работы и тестирования системы </b>
Результат распределения обучающих данных:
<img width="690" height="390" alt="image" src="https://github.com/user-attachments/assets/4720ce13-6f03-41b9-bb6b-3b57f3d5303b" />

<b> Процесс обучения </b>

Loss
<img width="547" height="418" alt="image" src="https://github.com/user-attachments/assets/8595211c-783b-4cb4-a550-53be3cf7de1a" />

Accuracy
<img width="547" height="418" alt="image" src="https://github.com/user-attachments/assets/047eceb1-ab59-4f52-9416-3c7c03f239ad" />

<b> Результат обучения </b>

                   precision    recall  f1-score   support

    Class 0 (Cat)       0.72      0.70      0.71        30
    Class 1 (Dog)       0.65      0.68      0.67        25

         accuracy                           0.69        55
        macro avg       0.69      0.69      0.69        55
     weighted avg       0.69      0.69      0.69        55

<b> Анализ метрик: </b>

Для класса 0 ("Кошки"):
1. Precision = 0.72: из всех изображений, которые модель предсказала как кошек, 72% действительно являются кошками;
2. Recall = 0.70: модель правильно идентифицировала 70% всех реальных кошек;
3. F1-score = 0.71: балансированная метрика качества классификации.

Для класса 1 ("Собаки"):
1. Precision = 0.65: из всех предсказанных собак, 65% действительно собаки;
2. Recall = 0.68: модель нашла 68% всех реальных собак;
3. F1-score = 0.67: немного ниже, чем для класса кошек.

<b> Матрица ошибок </b>

                  Кошка   Собака
    Кошка (30)       21       9
    Собака (25)       8      17

## <b> Выводы </b>
В ходе работы была обучена модель CNN.
Доля правильных ответов на обучающей/валидационной выборках: 0.9752/0.9432.
Вручную собранные данные модель размечает с общей долей правильных ответов: 0.69.
