"""Modul 8: normal equation. Dataset bawaan merupakan SIMULASI."""
from pathlib import Path
import csv
import io
import json
import numpy as np
import pandas as pd


def baca_csv(raw):
    try:
        text = raw.decode('utf-8-sig')
        sep = ';' if ';' in text.splitlines()[0] else ','
        rows = list(csv.reader(io.StringIO(text), delimiter=sep))
        if len(rows) < 2 or len(rows[0]) < 2:
            raise ValueError('CSV kosong atau kolom kurang.')
        if len(set(rows[0])) != len(rows[0]) or any(not h.strip() for h in rows[0]):
            raise ValueError('Nama kolom kosong atau berulang.')
        if any(len(row) != len(rows[0]) for row in rows[1:]):
            raise ValueError('Jumlah kolom CSV tidak konsisten.')
        return pd.read_csv(io.StringIO(text), sep=sep)
    except (UnicodeError, IndexError, csv.Error, pd.errors.ParserError) as exc:
        raise ValueError('CSV tidak valid.') from exc


def fit(df, fitur, target, intersep=True):
    if df.empty or not fitur or target in fitur or len(set(fitur)) != len(fitur):
        raise ValueError('Data kosong atau pemilihan fitur tidak valid.')
    try:
        values = df[fitur + [target]].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    except Exception as exc:
        raise ValueError('Kolom harus tersedia dan numerik.') from exc
    if not np.isfinite(values).all():
        raise ValueError('NaN/Inf/sel kosong tidak diperbolehkan.')
    X, y = values[:, :-1], values[:, -1]
    if intersep:
        X = np.column_stack([np.ones(len(X)), X])
    if len(X) < X.shape[1]:
        raise ValueError('Jumlah sampel kurang dari parameter.')
    with np.errstate(over='ignore', invalid='ignore'):
        XtX, Xty = X.T @ X, X.T @ y
    if not np.isfinite(XtX).all() or not np.isfinite(Xty).all():
        raise ValueError('Overflow matriks; perkecil skala data.')
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        raise ValueError('Matriks normal singular atau hampir singular.')
    w = np.linalg.solve(XtX, Xty)
    prediksi = X @ w
    residual = y - prediksi
    sse = float(residual @ residual)
    sst = float(np.sum((y-y.mean())**2))
    return dict(X=X, y=y, XtX=XtX, Xty=Xty, w=w, prediksi=prediksi,
                residual=residual, jumlah_galat=float(residual.sum()), sse=sse,
                rmse=float(np.sqrt(sse/len(y))), r2=None if sst == 0 else 1-sse/sst,
                ortogonal=X.T @ residual)


def proyeksi(u, a):
    u, a = np.asarray(u, dtype=float), np.asarray(a, dtype=float)
    if u.shape != a.shape or u.ndim != 1 or not np.isfinite([u,a]).all() or a @ a == 0:
        raise ValueError('Vektor harus finite, berdimensi sama, a bukan nol.')
    k = float((u @ a)/(a @ a))
    p = k*a
    return k, p, u-p


def contoh_3d():
    h = fit(pd.DataFrame({'x':[1,2,3], 'y':[1,3,4]}), ['x'], 'y')
    # P adalah vektor fitted di ruang observasi R3, bukan vektor koefisien R2.
    return h


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=str(Path(__file__).with_name('data_esteh_SIMULASI.csv')))
    parser.add_argument('--suhu', type=float, default=32.5)
    args = parser.parse_args()
    df = baca_csv(Path(args.csv).read_bytes())
    h = fit(df, ['suhu_udara_C'], 'penjualan_gelas_per_hari')
    out = {k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in h.items()}
    out['status_data'] = 'SIMULASI untuk CSV bawaan; bukan observasi lapangan'
    out['suhu_baru_C'] = args.suhu
    out['prediksi_baru_gelas'] = float(np.array([1,args.suhu]) @ h['w'])
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
