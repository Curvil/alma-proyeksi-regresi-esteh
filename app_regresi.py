from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from curve_fitting import baca_csv, fit, proyeksi, contoh_3d

st.set_page_config(page_title='Modul 8 | Suhu dan Es Teh', layout='wide')
st.title('Proyeksi Ortogonal dan Regresi Es Teh')
st.warning('DATA SIMULASI — 8 pasangan ilustratif, bukan pengukuran cuaca atau catatan penjualan nyata. Ganti dengan observasi sebelum mengklaim data lapangan.')
mode = st.radio('Materi', ['Regresi suhu dan penjualan', 'Proyeksi 2D', 'Bukti regresi 3D'])
if mode == 'Proyeksi 2D':
    ux = st.number_input('u x', value=1.)
    uy = st.number_input('u y', value=3.)
    ax = st.number_input('a x', value=5.)
    ay = st.number_input('a y', value=1.)
    try:
        k,p,e = proyeksi([ux,uy], [ax,ay])
        st.write({'k': k, 'P':p, 'e':e, 'e dot a':float(e @ [ax,ay])})
        fig, axis = plt.subplots(figsize=(6,5))
        for v, label in [(np.array([ux,uy]),'u'),(np.array([ax,ay]),'a'),(p,'p')]:
            axis.quiver(0,0,*v,angles='xy',scale_units='xy',scale=1,label=label)
        axis.plot([p[0],ux],[p[1],uy],'r--',label='e')
        lim=max(1,abs(ux),abs(uy),abs(ax),abs(ay))*1.3
        axis.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect='equal'); axis.grid();axis.legend()
        st.pyplot(fig);plt.close(fig)
    except ValueError as exc:
        st.error(str(exc))
    st.stop()
if mode == 'Bukti regresi 3D':
    h=contoh_3d()
    st.write('c1=(1,1,1); c2=(1,2,3); y=(1,3,4). P berada di span(c1,c2), e=y-P.')
    st.write({'w':h['w'], 'P':h['prediksi'], 'e':h['residual'], 'c1.e dan c2.e':h['ortogonal']})
    fig=plt.figure(figsize=(7,5)); ax=fig.add_subplot(projection='3d')
    s,t=np.meshgrid(np.linspace(-1,2,8),np.linspace(-1,2,8))
    ax.plot_surface(s+t,s+2*t,s+3*t,alpha=.15)
    for v,label in [(h['y'],'y'),(h['prediksi'],'P')]:
        ax.quiver(0,0,0,*v,label=label)
    p=h['prediksi'];y=h['y'];ax.plot([p[0],y[0]],[p[1],y[1]],[p[2],y[2]],label='e')
    ax.set(xlabel='Observasi 1',ylabel='Observasi 2',zlabel='Observasi 3');ax.legend()
    st.pyplot(fig);plt.close(fig);st.stop()
root=Path(__file__).resolve().parent
file=st.file_uploader('Unggah CSV pengamatan',type=['csv'])
contoh=st.checkbox('Gunakan DATA SIMULASI bawaan',value=True)
raw=file.getvalue() if file else (root/'data_esteh_SIMULASI.csv').read_bytes() if contoh else None
if raw is None:
    st.session_state.pop('hasil',None);st.info('Unggah CSV atau aktifkan simulasi.');st.stop()
try:
    df=baca_csv(raw)
except Exception as exc:
    st.session_state.pop('hasil',None);st.error(str(exc));st.stop()
st.dataframe(df,hide_index=True)
st.caption('File unggahan tidak diverifikasi sebagai observasi. Kolom sampel/status_data bukan fitur.')
cols=[c for c in df.columns if c.lower() not in ('sampel','no','id','nomor','status_data')]
if len(cols)<2:
    st.session_state.pop('hasil',None);st.error('Minimal dua kolom fitur/target.');st.stop()
digest=hashlib.sha256(raw).hexdigest()
target=st.selectbox('Target',cols,index=cols.index('penjualan_gelas_per_hari') if 'penjualan_gelas_per_hari' in cols else len(cols)-1,key='target'+digest)
choices=[c for c in cols if c!=target]
features=st.multiselect('Fitur',choices,default=choices,key='features'+digest+target)
intercept=st.checkbox('Intersep',value=True)
fp=hashlib.sha256(raw+json.dumps([features,target,intercept]).encode()).hexdigest()
if st.session_state.get('hasil',{}).get('fp')!=fp:
    st.session_state.pop('hasil',None)
if st.button('Hitung regresi',type='primary'):
    st.session_state.pop('hasil',None)
    try:
        h=fit(df,features,target,intercept);h['fp']=fp;st.session_state['hasil']=h
    except ValueError as exc:
        st.error(str(exc))
h=st.session_state.get('hasil')
if not h:st.stop()
st.subheader('Normal equation: (XᵀX)w = Xᵀy')
for label in ['X','y','XtX','Xty','w']:
    st.write(label);st.dataframe(h[label])
st.info('Model: '+target+' = '+' + '.join(f'{v:.8f}'+('' if n=='intersep' else ' × '+n) for n,v in zip((['intersep'] if intercept else [])+features,h['w'])))
for label in ['jumlah_galat','sse','rmse','r2']:
    st.write(label, h[label])
st.write('Xᵀe (bukti ortogonal)',h['ortogonal'])
st.caption('Dengan intersep, kolom 1 membuat Σe≈0; SSE=Σe² umumnya bukan nol. Angka mendekati nol dipengaruhi floating point. Evaluasi hanya data latih.')
tabel=pd.DataFrame({'aktual':h['y'],'prediksi':h['prediksi'],'residual':h['residual']})
st.dataframe(tabel,hide_index=True)
st.download_button('Unduh residual CSV',tabel.to_csv(index=False).encode(),'hasil_residual.csv','text/csv')
if len(features)==1:
    fig,axes=plt.subplots(1,2,figsize=(10,4));x=df[features[0]].to_numpy(dtype=float);idx=np.argsort(x)
    axes[0].scatter(x,h['y'],label='Data');axes[0].plot(x[idx],h['prediksi'][idx],label='OLS');axes[0].set(xlabel=features[0],ylabel=target);axes[0].legend()
    axes[1].scatter(h['prediksi'],h['residual']);axes[1].axhline(0,color='gray');axes[1].set(xlabel='Prediksi',ylabel='Residual')
    fig.tight_layout();st.pyplot(fig);plt.close(fig)
st.subheader('Prediksi baru')
values=[st.number_input('Nilai '+c,value=32.5 if c=='suhu_udara_C' else float(pd.to_numeric(df[c]).mean()),key=fp+c) for c in features]
x=np.array(([1.] if intercept else [])+values)
with np.errstate(over='ignore', invalid='ignore'):
    pred=float(x@h['w'])
if np.isfinite(pred):
    st.success(f'Prediksi {target} = {pred:.4f}')
    if pred<0:st.warning('Prediksi negatif tidak masuk akal untuk penjualan.')
    if any(v<float(df[c].min()) or v>float(df[c].max()) for c,v in zip(features,values)):
        st.warning('Ekstrapolasi di luar rentang pelatihan.')
else:st.error('Input prediksi tidak finite atau overflow.')
st.caption('Hubungan simulasi bukan bukti suhu menyebabkan penjualan; faktor lokasi, harga, hujan dan promosi belum diteliti.')
