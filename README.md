
## DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks

#### 1. Требования
- Python 3.11
- uv package manager
- TensorFlow 2.19
- Nvidia GPU

#### 2. Первые шаги
1. Для начала склонируйте этот репозиторий
```bash
git clone https://github.com/nnstu-miriteam/tensorflow-DPED
```
2. Установите менеджер пакетов [uv](https://docs.astral.sh/uv/getting-started/installation) для python в вашу систему.
3. Перейдите в директорию и создайте виртуальное окружение
```bash
cd tensorflow-DPED/
```
А затем установите необходимые пакеты командой
```bash
uv sync
```
5. Войдите или зарегистрируйтесь на [Weights & Biases (wandb)](https://wandb.ai/) для отслеживания вашего прогресса
6. Скачайте предобученную модель [VGG-19 model](https://polybox.ethz.ch/index.php/s/7z5bHNg5r5a0g7k) <sup>[Mirror](https://drive.google.com/file/d/0BwOLOmqkYj-jMGRwaUR2UjhSNDQ/view?usp=sharing&resourcekey=0-Ff-0HUQsoKJxZ84trhsHpA)</sup> и сохраните ее в директории `vgg_pretrained/`
- Скачайте по желанию оригинальный датасет [DPED dataset](http://people.ee.ethz.ch/~ihnatova/#dataset) и разрхивируйте его в папку `dped/`. Там должно быть три подпапки: `sony`, `iphone` и `blackberry`.   
7. Также можете скачать наш датасет и разместить его в папке `dped/`, там появится папка `kvadra`.
Ссылка для скачивания: https://huggingface.co/datasets/i44p/dped-pytorch/tree/main
Основные файлы:
- Обучающая выборка фотографий: `train.tar.zst`
- Тестовая выборка фотографий: `test.tar.zst`
- Патчи для обучения (~300k патчей): `train_patches.tar.zst` (собраны с фотографий обучающей выборки)
- Патчи для тестирования (~5k патчей): `test_patches.tar.zst` (собраны с фотографий тестовой выборки)
- Весь датасет (~1200 фотографий): `full_dataset_jpeg.tar.zst`
Извлеките `train_patches.tar.zst` и `test_patches.tar.zst` по пути `dped/kvadra/training_data/` и `dped/kvadra/test_data/patches/` соответсвенно.  
Переименуйте папку `target` в `sony` и папку `input` в `kvadra`.

#### 3. Обучение модели
```bash
uv run train_model.py model=<model> batch_size=<batch_size>
```
Необходимые параметры

>`model`: **`iphone`**, **`blackberry`**, **`sony`**, **`kvadra`**

Опциональные параметры

>```batch_size```: **```50```** &nbsp; - &nbsp; размер батча [меньшее значение может приводить к нестабильному обучению] <br/>
>```train_size```: **```30000```** &nbsp; - &nbsp; количество патчей, загруженных для обучения случайным образом каждый `eval_step`, который равен 1000 итерациям. Вы также можете загружать весь датасет вместо 30000 случайных каждый `eval_step`. Для этого введите `train_size=-1` в командой строке, но для этого потребуется много оперативной памяти<br/>
>```eval_step```: **```1000```** &nbsp; - &nbsp; каждые ```eval_step``` итераций модель считает метрики, сохраняет веса на текущем шаге и перезагружает датасет<br/>
>```num_train_iters```: **```20000```** &nbsp; - &nbsp; число тренировочных шагов <br/>
>```learning_rate```: **```5e-4```** &nbsp; - &nbsp; скорость обучения <br/>
>```w_content```: **```10```** &nbsp; - &nbsp; коэффициент функции потерь по содержанию<br/>
>```w_color```: **```0.5```** &nbsp; - &nbsp; коэффициент функции потерь по цвету <br/>
>```w_texture```: **```1```** &nbsp; - &nbsp; коэффициент функции потерь по текстуре (потерь дискриминатора) <br/>
>```w_tv```: **```2000```** &nbsp; - &nbsp; коэффициент функции потерь по снижению шума<br/>
>```dped_dir```: **```dped/```** &nbsp; - &nbsp; путь к директории с датасетом<br/>
>```vgg_dir```: **```vgg_pretrained/imagenet-vgg-verydeep-19.mat```** &nbsp; - &nbsp; путь до директории с предобученной моделью VGG-19 <br/>
>```gpu```: &nbsp; - &nbsp; порядковый номер видеокарты, если у вас несколько графических процессоров<br/>

Пример:

```bash
uv run train_model.py model=kvadra batch_size=50 dped_dir=dped/
```

#### 4. тестирование полученной модели

```bash
uv test_model.py model=<model>
```

Необходимые параметры

>```model```: **```iphone```**, **```blackberry```** , **```sony```**, **`kvadra`**

Опциональные параметры:

>```test_subset```: **```full```**,**```small```**  &nbsp; - &nbsp; all 29 or only 5 test images will be processed <br/>
>```iteration```: **```all```** or **```<number>```**  &nbsp; - &nbsp; обработать фотографии на всех итерациях или определенной итерации (1000, 2000, ..., 19000)
>```resolution```: **```orig```**,**```high```**,**```medium```**,**```small```**,**```tiny```** &nbsp; - &nbsp; разрешение выходных изображений после обработки
Оригинальные разрешения устройств:
Kvadra 4224x3136
Iphone
Sony
BlackBerry
>```use_gpu```: **```true```**,**```false```** &nbsp; - &nbsp; запустить обработку с использованием графического процессора <br/>
>```dped_dir```: **```dped/```** &nbsp; - &nbsp; путь до директории с датасетом <br/>  

Пример:

```bash
uv run test_model.py model=kvadra iteration=19000 test_subset=full resolution=orig use_gpu=true 
```
Для фотографий с высоким разрешением (к примеру 4224×3136), TensorFlow может упасть с ошибкой в нехватке памяти. В таком случае используйте `use_gpu=false`.
Скрипт `test_model.py` обрабатывает фотографии, хранящиеся в директории
`dped/kvadra/test_data/full_size_test_images/`
Для обработки одного изображения используйте скрипт `test_image.py`.
Модель, обученная для планшета Kvadra, поддерживает разрешение 4224x3136, поэтому с умом подходите к выбору фотографии для обработки. Все разрешения хранятся в файле `utils.py`.
Пример:
```bash
uv run test_image.py <path_to_image> --iter 18000 --gpu true
```
По умолчанию iteration=19000
По умолчанию gpu=false
<br/>

#### 5. Структура проекта

>```dped/```              &nbsp; - &nbsp; директория, в которой хранится датасет <br/>
>```models/```            &nbsp; - &nbsp; логи и модели сохраняются в этой папке в процессе обучения <br/>
>```models_orig/```       &nbsp; - &nbsp; предобученные модели **`sony`**, **`iphone`**, **`blackberry`** <br/>
>```results/```           &nbsp; - &nbsp; обработка нескольких патчей в процессе обучения при сохранении модели <br/>
>```vgg-pretrained/```    &nbsp; - &nbsp; директория с моделью VGG-19 <br/>
>```visual_results/```    &nbsp; - &nbsp; папка с обработанными фотографиями после скрипта `test_model.py`<br/>

>```load_dataset.py```    &nbsp; - &nbsp; скрипт для загрузки датасета для обучения <br/>
>```models.py```          &nbsp; - &nbsp; архитектура генератора и дискриминатора <br/>
>```ssim.py```            &nbsp; - &nbsp; реализация ssim метрики <br/>
>```train_model.py```     &nbsp; - &nbsp; скрипт для обучения модели <br/>
>```test_model.py```      &nbsp; - &nbsp; скрипт для тестирования модели <br/>
>```test_image.py```      &nbsp; - &nbsp; скрипт для тестирования модели для одного изображения<br/>
>```utils.py```           &nbsp; - &nbsp; вспомогательные функции <br/>
>```vgg.py```             &nbsp; - &nbsp; загрузка предобученной модели VGG-19 <br/>

