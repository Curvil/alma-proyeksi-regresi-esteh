# Proyeksi Ortogonal dan Regresi Es Teh

Aplikasi Streamlit pembelajaran proyeksi 2D, regresi linear dengan normal equation, dan proyeksi regresi di ruang observasi 3D.

**DATA SIMULASI:** `data_esteh_SIMULASI.csv` berisi 8 pasangan ilustratif suhu dan penjualan, bukan pengukuran cuaca atau catatan penjualan nyata. Hasil model hanya evaluasi data latih; tidak membuktikan hubungan sebab-akibat atau kelayakan operasional.

## Jalankan

Gunakan Python 3.11 atau 3.12.

```sh
pip install -r requirements.txt
streamlit run app_regresi.py
```

CSV pengamatan dapat diunggah melalui aplikasi. Pilih fitur numerik dan target. Kolom `sampel` dan `status_data` bukan fitur. Jangan unggah data pribadi atau rahasia.

## Streamlit Community Cloud

- Branch: `main`
- Main file path: `app_regresi.py`
- Dependencies: `requirements.txt`
- Tidak memerlukan secrets.

Paket publik hanya memuat kode aplikasi, dependensi, dataset SIMULASI, dan README. Laporan, identitas mahasiswa, NIM, tangkapan layar, dan riwayat repositori kuliah tidak disertakan.
