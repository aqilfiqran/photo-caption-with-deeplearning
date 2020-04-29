# photo-caption-with-deeplearning
	Ini merupakan mesin yang memprediksi tentang isi dari gambar dalam bentuk teks yang dibuat menggunakan bantuan tensorflow 

## Run
1. Unduh terlebih dahulu gambar yang akan dijadikan dataset [disini](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
2. Unduh juga text deskripsi dari dataset gambar diatas [disini](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)
3. Extract data yang telah diunduh
4. Install modul python dengan sintaks
```
    pip install -r requirements.txt
```
5. Buatlah fitur dari dataset, deskripsinya dan tokenizer
```
    $ python create_feature.py
```
6. Buatlah folder model pada di root projek
7. Lakukan training pada fitur
```
    $ python train.py
```
8. Lakukan testing data 
```
    $ python test.py
```
9. Output yang akan dihasilkan kurang lebih seperti ini
```
    startseq dog is running across the beach endseq
```

## Directory Structure
```bash
├── Flicker8k_Dataset
│   └── *.jpg
├── Flicker8k_text
│   └── *.txt
├── model
│   └── *.h5
├── example.jpg
└── *.py
```

## Requirement
- [Python](https://www.python.org/)
- [Tensorflow](https://www.tensorflow.org/install/source)

## Source
- [Generate Caption](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)
