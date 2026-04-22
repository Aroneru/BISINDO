# BISINDO Model Runner

Aplikasi Python untuk menjalankan model `Bisindo.pth` pada gambar atau kamera secara realtime.

## Kebutuhan

- Python 3.10 atau lebih baru
- Paket Python:

```bash
pip install torch torchvision pillow numpy opencv-python
```

Jika ingin menjalankan di GPU, pastikan versi `torch` yang dipasang mendukung CUDA di perangkat Anda.

## Struktur File

- `use_bisindo_model.py` - script utama untuk inferensi
- `Bisindo.pth` - model utama
- `checkpoint.pth` - checkpoint tambahan
- `labels.txt` - opsional, daftar label kelas satu baris satu label

## Cara Menjalankan

### 1. Mode Kamera Realtime

Mode ini akan membuka webcam dan menampilkan prediksi langsung di jendela video.

```bash
python use_bisindo_model.py --model-path Bisindo.pth --camera
```

Opsi tambahan:

```bash
python use_bisindo_model.py --model-path Bisindo.pth --camera --camera-id 0 --topk 3 --infer-every-n-frames 3
```

Keterangan:

- `--camera-id 0` menggunakan kamera default
- `--topk 3` menampilkan 3 prediksi teratas
- `--infer-every-n-frames 3` menjalankan inferensi setiap 3 frame agar lebih ringan

Tekan `q` untuk keluar dari jendela kamera.

### 2. Mode Gambar Tunggal

Jika ingin menguji satu gambar saja:

```bash
python use_bisindo_model.py --model-path Bisindo.pth --image contoh.jpg
```

### 3. Jika Ada File Label

Jika Anda punya daftar label kelas, misalnya `labels.txt`:

```bash
python use_bisindo_model.py --model-path Bisindo.pth --camera --labels labels.txt
```

atau:

```bash
python use_bisindo_model.py --model-path Bisindo.pth --image contoh.jpg --labels labels.txt
```

## Jika Model Perlu Arsitektur Manual

Kalau checkpoint Anda tidak bisa dideteksi otomatis, tambahkan arsitektur model asli dengan parameter berikut:

```bash
python use_bisindo_model.py --model-path Bisindo.pth --camera --model-module nama_file_model --model-class NamaClassModel --num-classes 26
```

## Catatan

- Script ini sudah mendukung mode kamera realtime secara default jika `--image` tidak diisi.
- File `.pth` disimpan dengan Git LFS karena ukurannya besar.
- Jika `torch` belum terpasang, instal dulu sebelum menjalankan aplikasi.
