
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

# Ниже сохранено оригинальное README

## DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks

<br/>

<img src="http://people.ee.ethz.ch/~ihnatova/assets/img/teaser_git.jpg"/>

<br/>

#### 1. Overview [[Paper]](https://arxiv.org/pdf/1704.02470.pdf) [[Project webpage]](http://people.ee.ethz.ch/~ihnatova/) [[Enhancing RAW photos]](https://github.com/aiff22/PyNET) [[Rendering Bokeh Effect]](https://github.com/aiff22/PyNET-Bokeh)

The provided code implements the paper that presents an end-to-end deep learning approach for translating ordinary photos from smartphones into DSLR-quality images. The learned model can be applied to photos of arbitrary resolution, while the methodology itself is generalized to 
any type of digital camera. More visual results can be found [here](http://people.ee.ethz.ch/~ihnatova/#demo).


#### 2. Prerequisites

- Python + Pillow, scipy, numpy, imageio packages
- [TensorFlow 1.x / 2.x](https://www.tensorflow.org/install/) + [CUDA CuDNN](https://developer.nvidia.com/cudnn)
- Nvidia GPU


#### 3. First steps

- Download the pre-trained [VGG-19 model](https://polybox.ethz.ch/index.php/s/7z5bHNg5r5a0g7k) <sup>[Mirror](https://drive.google.com/file/d/0BwOLOmqkYj-jMGRwaUR2UjhSNDQ/view?usp=sharing&resourcekey=0-Ff-0HUQsoKJxZ84trhsHpA)</sup> and put it into `vgg_pretrained/` folder
- Download [DPED dataset](http://people.ee.ethz.ch/~ihnatova/#dataset) (patches for CNN training) and extract it into `dped/` folder.  
<sub>This folder should contain three subolders: `sony/`, `iphone/` and `blackberry/`</sub>

<br/>

#### 4. Train the model

```bash
python train_model.py model=<model>
```

Obligatory parameters:

>```model```: **```iphone```**, **```blackberry```** or **```sony```**

Optional parameters and their default values:

>```batch_size```: **```50```** &nbsp; - &nbsp; batch size [smaller values can lead to unstable training] <br/>
>```train_size```: **```30000```** &nbsp; - &nbsp; the number of training patches randomly loaded each ```eval_step``` iterations <br/>
>```eval_step```: **```1000```** &nbsp; - &nbsp; each ```eval_step``` iterations the model is saved and the training data is reloaded <br/>
>```num_train_iters```: **```20000```** &nbsp; - &nbsp; the number of training iterations <br/>
>```learning_rate```: **```5e-4```** &nbsp; - &nbsp; learning rate <br/>
>```w_content```: **```10```** &nbsp; - &nbsp; the weight of the content loss <br/>
>```w_color```: **```0.5```** &nbsp; - &nbsp; the weight of the color loss <br/>
>```w_texture```: **```1```** &nbsp; - &nbsp; the weight of the texture [adversarial] loss <br/>
>```w_tv```: **```2000```** &nbsp; - &nbsp; the weight of the total variation loss <br/>
>```dped_dir```: **```dped/```** &nbsp; - &nbsp; path to the folder with DPED dataset <br/>
>```vgg_dir```: **```vgg_pretrained/imagenet-vgg-verydeep-19.mat```** &nbsp; - &nbsp; path to the pre-trained VGG-19 network <br/>

Example:

```bash
python train_model.py model=iphone batch_size=50 dped_dir=dped/ w_color=0.7
```

<br/>

#### 5. Test the provided pre-trained models

```bash
python test_model.py model=<model>
```

Obligatory parameters:

>```model```: **```iphone_orig```**, **```blackberry_orig```** or **```sony_orig```**

Optional parameters:

>```test_subset```: **```full```**,**```small```**  &nbsp; - &nbsp; all 29 or only 5 test images will be processed <br/>
>```resolution```: **```orig```**,**```high```**,**```medium```**,**```small```**,**```tiny```** &nbsp; - &nbsp; the resolution of the test images [**```orig```** means original resolution]<br/>
>```use_gpu```: **```true```**,**```false```** &nbsp; - &nbsp; run models on GPU or CPU <br/>
>```dped_dir```: **```dped/```** &nbsp; - &nbsp; path to the folder with DPED dataset <br/>

Example:

```bash
python test_model.py model=iphone_orig test_subset=full resolution=orig use_gpu=true
```

<br/>

#### 6. Test the obtained models

```bash
python test_model.py model=<model>
```

Obligatory parameters:

>```model```: **```iphone```**, **```blackberry```** or **```sony```**

Optional parameters:

>```test_subset```: **```full```**,**```small```**  &nbsp; - &nbsp; all 29 or only 5 test images will be processed <br/>
>```iteration```: **```all```** or **```<number>```**  &nbsp; - &nbsp; get visual results for all iterations or for the specific iteration,  
>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**```<number>```** must be a multiple of ```eval_step``` <br/>
>```resolution```: **```orig```**,**```high```**,**```medium```**,**```small```**,**```tiny```** &nbsp; - &nbsp; the resolution of the test 
images [**```orig```** means original resolution]<br/>
>```use_gpu```: **```true```**,**```false```** &nbsp; - &nbsp; run models on GPU or CPU <br/>
>```dped_dir```: **```dped/```** &nbsp; - &nbsp; path to the folder with DPED dataset <br/>  

Example:

```bash
python test_model.py model=iphone iteration=13000 test_subset=full resolution=orig use_gpu=true
```
<br/>

#### 7. Folder structure

>```dped/```              &nbsp; - &nbsp; the folder with the DPED dataset <br/>
>```models/```            &nbsp; - &nbsp; logs and models that are saved during the training process <br/>
>```models_orig/```       &nbsp; - &nbsp; the provided pre-trained models for **```iphone```**, **```sony```** and **```blackberry```** <br/>
>```results/```           &nbsp; - &nbsp; visual results for small image patches that are saved while training <br/>
>```vgg-pretrained/```    &nbsp; - &nbsp; the folder with the pre-trained VGG-19 network <br/>
>```visual_results/```    &nbsp; - &nbsp; processed [enhanced] test images <br/>

>```load_dataset.py```    &nbsp; - &nbsp; python script that loads training data <br/>
>```models.py```          &nbsp; - &nbsp; architecture of the image enhancement [resnet] and adversarial networks <br/>
>```ssim.py```            &nbsp; - &nbsp; implementation of the ssim score <br/>
>```train_model.py```     &nbsp; - &nbsp; implementation of the training procedure <br/>
>```test_model.py```      &nbsp; - &nbsp; applying the pre-trained models to test images <br/>
>```utils.py```           &nbsp; - &nbsp; auxiliary functions <br/>
>```vgg.py```             &nbsp; - &nbsp; loading the pre-trained vgg-19 network <br/>

<br/>

#### 8. Problems and errors

```
What if I get an error: "OOM when allocating tensor with shape [...]"?
```

&nbsp;&nbsp; Your GPU does not have enough memory. If this happens during the training process:

- Decrease the size of the training batch [```batch_size```]. Note however that smaller values can lead to unstable training.

&nbsp;&nbsp; If this happens while testing the models:

- Run the model on CPU (set the parameter ```use_gpu``` to **```false```**). Note that this can take up to 5 minutes per image. <br/>
- Use cropped images, set the parameter ```resolution``` to:

> **```high```**   &nbsp; - &nbsp; center crop of size ```1680x1260``` pixels <br/>
> **```medium```** &nbsp; - &nbsp; center crop of size ```1366x1024``` pixels <br/>
> **```small```** &nbsp; - &nbsp; center crop of size ```1024x768``` pixels <br/>
> **```tiny```** &nbsp; - &nbsp; center crop of size ```800x600``` pixels <br/>

&emsp;&nbsp; The less resolution is - the smaller part of the image will be processed

<br/>

#### 9. Citation

```
@inproceedings{ignatov2017dslr,
  title={DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks},
  author={Ignatov, Andrey and Kobyshev, Nikolay and Timofte, Radu and Vanhoey, Kenneth and Van Gool, Luc},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3277--3285},
  year={2017}
}
```


#### 10. Any further questions?

```
Please contact Andrey Ignatov (andrey.ignatoff@gmail.com) for more information
```
