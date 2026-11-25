import os, re, math, unicodedata, zipfile
from pathlib import Path
from itertools import combinations
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from IPython.display import display

dosya_adi=os.environ.get('APR_EXCEL', '').strip()
if not dosya_adi:
    try:
        from google.colab import files
        yuklenen=files.upload()
        if not yuklenen: raise ValueError('Özgün Excel dosyası yüklenmedi.')
        dosya_adi=next(iter(yuklenen))
    except ImportError:
        matches=list(Path('.').glob('Behavior_Analysis_for_university_ Students*.xlsx'))
        if not matches: raise FileNotFoundError('53 sütunlu özgün İngilizce Excel dosyasını çalışma klasörüne koyun.')
        dosya_adi=str(matches[0])

excel=pd.ExcelFile(dosya_adi)
df_raw=pd.read_excel(dosya_adi,sheet_name=excel.sheet_names[0],dtype=object,keep_default_na=False)
df_raw=df_raw.loc[~df_raw.apply(lambda r: all(str(v).strip()=='' for v in r),axis=1)].reset_index(drop=True)
df_raw.columns=[str(x).strip() for x in df_raw.columns]
if df_raw.shape != (351,53) or not df_raw.columns[0].startswith('Semester('):
    raise ValueError(f'351x53 özgün Mendeley V2 veri dosyası bekleniyor, alınan: {df_raw.shape}')
print('Özgün dosya:',dosya_adi,'| satır, sütun:',df_raw.shape)

def txt(value):
    if value is None or pd.isna(value):
        return ''
    value = unicodedata.normalize('NFKC', str(value))
    return re.sub(r'\s+', ' ', value).strip().casefold()

EKSIK = {'', 'n/a', 'na', 'null', 'none', '-', 'nan'}
def eksik(value):
    return txt(value) in EKSIK

KARDES_IFADE = {
    '1 big sister':1, '1 elder sister':1, '1 sister.':1,
    '1 brother 1 sister':2, '2 sister':2,
    '3 brothers':3, '3 sisters':3,
    'numbers: 1':1,
}

def kardeş_sayisi(value):
    s = txt(value)
    if s in {'none', 'no', 'no one'}:
        return 0
    if s in KARDES_IFADE:
        return KARDES_IFADE[s]
    if s == '10+':
        return 10
    if re.fullmatch(r'0*\d{1,3}', s):
        num = int(s)
        return num if num <= 30 else np.nan
    return np.nan

sib_col = df_raw.columns[9]
hatali = df_raw[sib_col].map(lambda x: txt(x).lstrip('0').isdigit() and int(txt(x)) > 1000)
print('Geçersiz kardeş sayısı nedeniyle dışlanan kayıt sayısı:', int(hatali.sum()))
print('Dışlanan özgün Excel satır numarası (başlık satırı +1):', list(df_raw.index[hatali] + 2))
assert int(hatali.sum()) == 1, 'Beklenen tek sıra dışı kayıt bulunamadı; dosya sürümünü kontrol edin.'
df = df_raw.loc[~hatali].copy().reset_index(drop=True)
assert len(df) == 350, 'Temizleme sonrası 350 kayıt bekleniyor.'

DASS_CEVAP = {
    'did not apply to me at all': 0,
    'applied to me to some degree, or some of the time': 1,
    'applied to me to a considerable degree or a good part of time': 2,
    'applied to me very much or most of the time': 3,
}
dass_kod = pd.DataFrame(index=df.index)
for q in range(1, 22):
    kolon = df.columns[31 + q]
    kod = df[kolon].map(lambda v: np.nan if eksik(v) else DASS_CEVAP.get(txt(v), np.nan))
    beklenmeyen = df.loc[df[kolon].map(lambda v: not eksik(v)) & kod.isna(), kolon].unique()
    if len(beklenmeyen):
        raise ValueError(f'DASS madde {q} için beklenmeyen yanıt(lar): {beklenmeyen}')
    dass_kod[f'DASS_{q:02d}'] = kod.astype(float)

print('Nefes darlığı (DASS madde 4) eksik:', int(dass_kod['DASS_04'].isna().sum()))
assert int(dass_kod['DASS_04'].isna().sum()) == 8
print('DASS madde düzeyinde eksik hücre sayısı:', int(dass_kod.isna().sum().sum()))
print('Uyku memnuniyeti — özgün 5 kategori:')
print(df.iloc[:,28].map(txt).value_counts().to_string())
print('Uyku sorununun fark edilmesi — özgün 5 kategori:')
print(df.iloc[:,30].map(txt).value_counts().to_string())

AD = {
  1:'Yaş', 4:'Cinsiyet', 5:'Din', 6:'Daimi_ikamet', 7:'Mevcut_ikamet',
  8:'Aile_tipi', 9:'Kardeş_sayısı', 10:'Baba_mesleği', 11:'Anne_mesleği',
  12:'Aile_ortamı', 13:'CGPA', 14:'Günlük_çalışma', 15:'Sosyal_medya',
  16:'İlişki_durumu', 17:'Sigara', 18:'Dini_uygulama', 19:'Akademik_memnuniyet',
  20:'Ders_dışı_katılım', 22:'Kariyer_fırsatı', 23:'Mesleğin_sosyal_değeri',
  24:'İntihar_madde_25', 25:'İntihar_madde_26', 26:'İntihar_madde_27',
  27:'İntihar_madde_28', 28:'Uyku_memnuniyeti', 29:'Uyku_işlev',
  30:'Uyku_farkedilme', 31:'Uyku_endişe',
}

UYKU_MEMNUN = {
    'very satisfied':'Çok_memnun', 'satisfied':'Memnun',
    'moderately satisfied':'Orta', 'dissatisfied':'Memnuniyetsiz',
    'very dissatisfied':'Çok_memnuniyetsiz',
}
UYKU_FARK = {
    'not at all noticeable':'Hiç_fark_edilmez', 'barely':'Zor_fark_edilir',
    'somewhat':'Kısmen', 'much':'Fazla', 'very much noticeable':'Çok_fazla',
}
UYKU_ISLEV = {
    'not at all interfering':'Hiç_etkilemiyor', 'a little':'Biraz',
    'somewhat':'Kısmen', 'much':'Fazla', 'very much interfering':'Çok_fazla',
}
UYKU_ENDISE = {
    'not at all':'Hiç', 'barely':'Çok_az', 'somewhat':'Kısmen',
    'much':'Fazla', 'very much':'Çok_fazla',
}
KATEGORI_HARITA = {28:UYKU_MEMNUN,29:UYKU_ISLEV,30:UYKU_FARK,31:UYKU_ENDISE}

def cgpa_kategori(value):
    s = txt(value).replace(',', '.')

    match = re.fullmatch(r'(?:cgpa\s*[-:]?\s*)?(\d(?:\.\d{1,2})?)', s)
    if not match:
        return None
    x = float(match.group(1))
    if not (0 < x <= 4): return None
    return '0-2.00' if x <= 2 else ('2.01-3.00' if x <= 3 else ('3.01-3.50' if x <= 3.5 else '3.51-4.00'))

def yas_araligi(value):
    s = txt(value)
    if s in {'19 years 11 months 24 days'}: return '20_ve_altı'
    if s in {'20 years 9 months 30 days'}: return '20_ve_altı'
    if s == '22.5': return '21-22'
    if s == '24+': return '23_ve_üstü'
    if s == 'years: 25': return '23_ve_üstü'
    if not re.fullmatch(r'\d{2}', s): return None
    n = int(s)
    if not 16 <= n <= 80: return None
    return '20_ve_altı' if n <= 20 else ('21-22' if n <= 22 else '23_ve_üstü')

def deger_kodla(i, v):
    if eksik(v) and not (i==9 and txt(v)=='none'):
        return None
    s = txt(v)
    if i == 1:
        kategori = yas_araligi(v)
        if kategori is None: return None
    elif i == 9:
        n = kardeş_sayisi(v)
        if pd.isna(n): return None
        kategori = '0' if n == 0 else ('1' if n == 1 else ('2' if n == 2 else '3_ve_üstü'))
    elif i == 13:
        kategori = cgpa_kategori(v)
        if kategori is None: return None
    elif i in KATEGORI_HARITA:
        kategori = KATEGORI_HARITA[i].get(s)
        if kategori is None:
            raise ValueError(f'Beklenmeyen uyku yanıtı (sütun {i}, değer={v!r})')
    else:
        kategori = {'yes':'Evet','no':'Hayır','neural':'Neutral','offen':'Often',
                    'strongly_disagree':'Strongly-disagree','dis-agree':'Disagree',
                    'join or extended':'Joint_or_Extended'}.get(s, re.sub(r'\s+', '_', s))
    return f'{AD[i]}={kategori}'

DEGISKEN_SAYISI = len(AD) + 21
assert DEGISKEN_SAYISI == 49
print('Madde düzeyi analize alınabilen soru sayısı:', DEGISKEN_SAYISI,
      '(53 özgün - Dönem, Boy, Kilo, açık uçlu etkinlik açıklaması)')
KAYIT_ITEMS = []
KOD_KAYIPLARI = []
for r, (_, row) in enumerate(df.iterrows()):
    values = {}
    for i in AD:
        coded = deger_kodla(i, row.iloc[i])
        if coded is None and not eksik(row.iloc[i]):
            KOD_KAYIPLARI.append({'excel_satiri':r+2, 'soru_no':i+1,
                                  'değişken':AD[i],'özgün_değer':str(row.iloc[i])})
        if coded is not None: values[i] = coded
    for j in range(1,22):
        val = dass_kod.loc[r, f'DASS_{j:02d}']
        if not pd.isna(val): values[31+j] = f'DASS_{j:02d}={int(val)}'
    KAYIT_ITEMS.append(values)

kod_kayiplari = pd.DataFrame(KOD_KAYIPLARI)
print('Yorumlanamayan sayısal/uyku yanıtı sayısı:',len(kod_kayiplari))
if not kod_kayiplari.empty:
    display(kod_kayiplari.groupby(['değişken','özgün_değer']).size().reset_index(name='n').head(35))

assert (df.iloc[:,28].map(txt)=='satisfied').sum() == 96
assert (df.iloc[:,30].map(txt)=='not at all noticeable').sum() == 127
assert (dass_kod['DASS_04']==0).sum() == 193
print('KONTROL BAŞARILI: Memnun=96, Hiç fark edilmez=127, nefes darlığı puan 0=193.')

ALT_OLCEK = {
    'Depresyon': [3,5,10,13,16,17,21],
    'Anksiyete': [2,4,7,9,15,19,20],
    'Stres': [1,6,8,11,12,14,18],
}

def cronbach_alpha(items):
    clean = items.dropna(axis=0, how='any').astype(float)
    k = clean.shape[1]
    total_var = clean.sum(axis=1).var(ddof=1)
    alpha = k/(k-1) * (1-clean.var(ddof=1).sum()/total_var) if total_var>0 else np.nan
    return len(clean), float(alpha)

P = pd.DataFrame(index=df.index)
guv = []
for ad, sayilar in ALT_OLCEK.items():
    alt = dass_kod[[f'DASS_{x:02d}' for x in sayilar]]
    P[ad] = alt.sum(axis=1, min_count=7)
    n, a = cronbach_alpha(alt)
    guv.append({'Alt_ölçek':ad, 'N_geçerli':n, 'Cronbach_alpha':round(a,5)})
n, a = cronbach_alpha(dass_kod)
guv.append({'Alt_ölçek':'DASS21_tüm_maddeler','N_geçerli':n,'Cronbach_alpha':round(a,5)})
guv_df = pd.DataFrame(guv)
display(guv_df)
assert [int(x) for x in guv_df.N_geçerli] == [350,342,350,342]

NORMAL_UST_SINIR_42 = {'Depresyon':9, 'Anksiyete':7, 'Stres':14}
for sub in ALT_OLCEK:
    toplam42 = P[sub]*2
    P[sub+'_Kategori'] = toplam42.map(lambda v: None if pd.isna(v) else
                                      ('Referans_üstü' if v>NORMAL_UST_SINIR_42[sub] else 'Referans_aralığı'))
ESIK_5 = {
    'Depresyon': [9,13,20,27],
    'Anksiyete': [7,9,14,19],
    'Stres': [14,18,25,33],
}
BES_ETIKET = ['Referans','Hafif','Orta','Yuksek','Cok_yuksek']
for sub, sinirlar in ESIK_5.items():
    def besli(v, s=sinirlar):
        if pd.isna(v): return None
        x = v*2
        return BES_ETIKET[next((k for k,t in enumerate(s) if x<=t),4)]
    P[sub+'_5_Kategori'] = P[sub].map(besli)
print('DASS-21 kategorilerinin dağılımı (geçerli vakalar):')
for sub in ALT_OLCEK:
    print(sub, P[sub+'_Kategori'].value_counts(dropna=False).to_dict())

GRUPLAR = {
 'Akademik': {'CGPA','Günlük_çalışma','Akademik_memnuniyet','Kariyer_fırsatı','Mesleğin_sosyal_değeri'},
 'Sosyal': {'Sosyal_medya','Ders_dışı_katılım','İlişki_durumu','Sigara','Dini_uygulama'},
 'Aile_ve_yasam': {'Daimi_ikamet','Mevcut_ikamet','Aile_tipi','Kardeş_sayısı','Baba_mesleği','Anne_mesleği','Aile_ortamı'},
 'Demografik': {'Yaş','Cinsiyet','Din'},
 'Psikososyal_diger': {'İntihar_madde_25','İntihar_madde_26','İntihar_madde_27','İntihar_madde_28',
                       'Uyku_memnuniyeti','Uyku_işlev','Uyku_farkedilme','Uyku_endişe'},
 'DASS21_madde': {f'DASS_{i:02d}' for i in range(1,22)},
}
GRUP_MAP = {x:k for k,s in GRUPLAR.items() for x in s}
DASS_TO_SUB={f'DASS_{j:02d}':sub for sub,items in ALT_OLCEK.items() for j in items}
assert set(GRUP_MAP)==set(AD.values())|{f'DASS_{i:02d}' for i in range(1,22)}, 'Gruplara atanmayan değişken var.'
PSY_GRUP={'DASS21_madde','Psikososyal_diger'}
OTHER_GRUP={'Akademik','Sosyal','Aile_ve_yasam','Demografik'}

def var(item): return item.split('=',1)[0]
def domain(item): return GRUP_MAP.get(var(item),'Bilinmeyen')
def maskbits(n): return (1<<n)-1

def frequency_masks(baskets):
    masks=defaultdict(int)
    for i,b in enumerate(baskets):
        for item in b: masks[item] |= (1<<i)
    return dict(masks)

def rules_from_masks(masks, N, support_floor=.10, prevalence_cut=.85, restrict_outcomes=None):
    need=math.ceil(support_floor*N-1e-10)
    useful={k:v for k,v in masks.items() if need<=v.bit_count() and v.bit_count()/N<=prevalence_cut}
    ordered=sorted(useful)
    pair={}
    for a,b in combinations(ordered,2):
        if var(a)==var(b): continue
        common=useful[a]&useful[b]
        if common.bit_count()>=need: pair[(a,b)]=common
    neighbors=defaultdict(set)
    for a,b in pair: neighbors[a].add(b)
    triple={}
    for (a,b),mask_ab in pair.items():
        for c in neighbors[a].intersection(neighbors[b]):
            if b>=c or var(a)==var(c) or var(b)==var(c): continue
            v=mask_ab & useful[c]
            if v.bit_count()>=need: triple[(a,b,c)]=v

    C=[c for c in ordered if restrict_outcomes is None or restrict_outcomes(c)]
    tested_ante=list((x,) for x in ordered)+list(pair)
    m=sum(1 for A in tested_ante for c in C if var(c) not in {var(x) for x in A})
    return useful,pair,triple,C,m

def ci_wilson(k,n,z=1.959963984540054):
    if n<=0:return (np.nan,np.nan)
    p=k/n; d=1+z*z/n; center=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+(z*z)/(4*n*n))/(d)
    return max(0,center-h),min(1,center+h)

def p_right(k, sample, population, success):

    if population<=0 or sample<=0: return np.nan
    return float(hypergeom.sf(k-1,population,success,sample))

def measure(A,c,maskAC,mask_dict,N):
    masksA=maskbits(N)
    for x in A: masksA &= mask_dict[x]
    nA=masksA.bit_count(); nC=mask_dict[c].bit_count(); nAC=maskAC.bit_count()
    if not (nA and nC): return None
    s=nAC/N; conf=nAC/nA; lift=conf/(nC/N)
    low,high=ci_wilson(nAC,nA)
    p=p_right(nAC,nA,N,nC)
    if len(A)==1:
        base_conf=nC/N; prod=conf-base_conf; p_prod=p
        compare='Sonuç temel yaygınlığına göre'; prod_rule=A[0]
    else:
        vals=[]; ps=[]
        for kept in A:
            nk=mask_dict[kept].bit_count()
            nkc=(mask_dict[kept]&mask_dict[c]).bit_count()
            vals.append(nkc/nk if nk else np.nan)

            if nk==nA: ps.append(1.0)
            else: ps.append(p_right(nAC,nA,nk,nkc))
        base_conf=max(vals); prod=conf-base_conf
        p_prod=max(ps)
        compare='İki tekil alt kurala göre'; prod_rule=' & '.join(A)
    return dict(N=N,nA=nA,nC=nC,nAC=nAC,Destek=s,Guven=conf,Lift=lift,
                GA95_alt=low,GA95_ust=high,Fisher_p=p,
                Alt_kural_en_yuksek_guven=base_conf,Uretkenlik_delta=prod,
                Uretkenlik_p=p_prod,Uretkenlik_karsilastirma=compare)

def class_rule(A,c):
    items=list(A)+[c]
    domains={domain(item) for item in items}
    haspsy=bool(domains & PSY_GRUP)
    nonpsy=domains & OTHER_GRUP
    if domains=={'DASS21_madde'}:
        involved={DASS_TO_SUB[var(item)] for item in items}
        return 'DASS_ayni_altolcek' if len(involved)==1 else 'DASS_altolcekler_arasi'
    if haspsy and nonpsy: return 'Alanlar_arasi'
    if haspsy: return 'Psikososyal_alan_ici'
    return 'Psikososyal_olmayan'

FULL_COLUMNS=['Model','Sinif','Oncul','Sonuc','Oncul_oge_sayisi','Oncul_alanlari','Sonuc_alani',
              'N','nA','nC','nAC','Destek','Guven','Lift','GA95_alt','GA95_ust','Fisher_p',
              'Global_Bonferroni_p','Alt_kural_en_yuksek_guven','Uretkenlik_delta',
              'Uretkenlik_p','Uretkenlik_Bonferroni_p','Uretkenlik_karsilastirma',
              'Uretkenlik_pozitif','Ek_katki_duzeltilmis_anlamli','Cevap_0_icerir']

def build_exploratory(fields_to_use, modelname):
    idx=[i for i in df.index if all(j in KAYIT_ITEMS[i] for j in fields_to_use)]
    baskets=[set(KAYIT_ITEMS[i][j] for j in fields_to_use) for i in idx]
    N=len(baskets); print(modelname,'Tam-vaka N:',N,'Değişken sayısı:',len(fields_to_use))
    bits=frequency_masks(baskets)
    used,p2,p3,C,m=rules_from_masks(bits,N,support_floor=.10,prevalence_cut=.85)
    print('Apriori öğekümeleri — sık 1:',len(used),'sık 2:',len(p2),'sık 3:',len(p3), 'Test ailesi:',m)
    records=[]
    for (a,b),common in p2.items():
        for A,c in [((a,),b),((b,),a)]:
            met=measure(A,c,common,used,N)
            if met['Guven']<.70-1e-12 or met['Lift']<1.20-1e-12:continue
            records.append((A,c,met))
    for (a,b,c),common in p3.items():
        for A,z in [((a,b),c),((a,c),b),((b,c),a)]:
            met=measure(A,z,common,used,N)
            if met['Guven']<.70-1e-12 or met['Lift']<1.20-1e-12:continue
            records.append((A,z,met))
    output=[]
    for A,c,met in records:
        groups={domain(x) for x in A}
        row=dict(Model=modelname,Sinif=class_rule(A,c),Oncul=' + '.join(A),Sonuc=c,
                 Oncul_oge_sayisi=len(A),Oncul_alanlari=';'.join(sorted(groups)),
                 Sonuc_alani=domain(c),**met)
        row['Global_Bonferroni_p']=min(1.,met['Fisher_p']*m)
        row['Uretkenlik_Bonferroni_p']=min(1.,met['Uretkenlik_p']*m)
        row['Uretkenlik_pozitif']=met['Uretkenlik_delta']>0
        row['Ek_katki_duzeltilmis_anlamli']=bool(met['Uretkenlik_delta']>0 and row['Uretkenlik_Bonferroni_p']<.05)
        row['Cevap_0_icerir']=any(domain(x)=='DASS21_madde' and x.endswith('=0') for x in list(A)+[c])
        output.append(row)
    frame=pd.DataFrame(output,columns=FULL_COLUMNS)
    frame=frame.sort_values(['Lift','Destek','Oncul','Sonuc'],ascending=[False,False,True,True]).reset_index(drop=True)
    diag={'Model':modelname,'N':N,'Degisken_sayisi':len(fields_to_use),'Sik_1_oge':len(used),
          'Sik_2_ogekumesi':len(p2),'Sik_3_ogekumesi':len(p3),
          'Global_Fisher_sinama_ailesi':m,'Kural_adet_en_gevsek_esik':len(frame)}
    return frame,diag

FIELDS_ALL=list(AD)+list(range(32,53))
assert len(FIELDS_ALL)==49
A_KURALLAR,A_AUDIT=build_exploratory(FIELDS_ALL,'A_49_degisken_tam_vaka')
A_SENS,A_SENS_AUDIT=build_exploratory([x for x in FIELDS_ALL if x not in {9,13}],
                                       'A_47_degisken_CGPA_kardes_haric')

def filter_cross(frame, categories, consequent_psy=True):
    rows=frame[frame.Sinif.eq('Alanlar_arasi')].copy()
    if consequent_psy:
        rows=rows[rows.Sonuc_alani.isin(PSY_GRUP)]
        rows=rows[rows.Oncul_alanlari.map(lambda s:any(x in s.split(';') for x in categories))]
    return rows

def main_threshold(frame, sup=.15, conf=.70, lift=1.40):
    return frame[(frame.Destek>=sup-1e-12)&(frame.Guven>=conf-1e-12)&(frame.Lift>=lift-1e-12)].copy()

def representative(rows, k=8, max_per_result=2):
    rows=rows.sort_values(['Lift','Destek','Oncul','Sonuc'],ascending=[False,False,True,True])
    picks=[]; cnt=Counter(); anteced=set()
    for _,r in rows.iterrows():
        if cnt[r.Sonuc]>=max_per_result or r.Oncul in anteced: continue
        picks.append(r.to_dict());cnt[r.Sonuc]+=1;anteced.add(r.Oncul)
        if len(picks)>=k:break
    return pd.DataFrame(picks,columns=FULL_COLUMNS)

A_ANA=main_threshold(A_KURALLAR)
A_TABLO4=representative(A_ANA[A_ANA.Sinif.isin(['DASS_ayni_altolcek','DASS_altolcekler_arasi','Psikososyal_alan_ici'])],10)
A_TABLO5=representative(filter_cross(A_ANA,{'Akademik','Sosyal'}),10)
A_TABLO6=representative(filter_cross(A_ANA,{'Aile_ve_yasam'}),10)
A_CROSS_ALL=filter_cross(A_KURALLAR,{'Akademik','Sosyal','Aile_ve_yasam','Demografik'})
A_TALIMATLAR=pd.DataFrame([
    {'Tablo':'4','Olcut':'Yalnız psikososyal ölçüme ait birliktelikler; DASS içi beklenen yapı ayrıca etiketlenir'},
    {'Tablo':'5','Olcut':'Akademik veya sosyal en az bir öncül + psikososyal sonuç; çapraz değişken katkısı ayrıca sınanır'},
    {'Tablo':'6','Olcut':'Ailesel/yaşam koşulu en az bir öncül + psikososyal sonuç; nedensel yorum yasak'},
    {'Tablo':'Hepsi','Olcut':'Önceden belirlenen eşikler: destek>=0,15; güven>=0,70; lift>=1,40; kategori yaygınlığı<=%85'},
    {'Tablo':'Hepsi','Olcut':'Temsilî seçim: lift azalan, destek azalan, aynı sonuç için en fazla iki; tüm kurallar ek dosyada'},
])
A_DUY=[]
for model, rules in [('49_degisken',A_KURALLAR),('CGPA_kardes_haric',A_SENS)]:
    for sup in [.10,.15,.20]:
        for lift in [1.20,1.40,1.60]:
            r=main_threshold(rules,sup=sup,lift=lift)
            cross=filter_cross(r,{'Akademik','Sosyal','Aile_ve_yasam','Demografik'})
            A_DUY.append(dict(Model=model,Destek=sup,Guven=.70,Lift=lift,
                              Tum_kural=len(r),DASS_ayni_altolcek=int(r.Sinif.eq('DASS_ayni_altolcek').sum()),
                              DASS_altolcekler_arasi=int(r.Sinif.eq('DASS_altolcekler_arasi').sum()),
                              Alanlar_arasi=len(cross),
                              Alanlar_arasi_duzeltilmis_fisher_lt_005=int(cross.Global_Bonferroni_p.lt(.05).sum()),
                              Iki_oncul_ek_katki_duzeltilmis_lt_005=int(cross[cross.Oncul_oge_sayisi.eq(2)].Ek_katki_duzeltilmis_anlamli.sum())))
A_DUY=pd.DataFrame(A_DUY)
print('ANA keşifsel — tüm kurallar:',len(A_ANA),'Tablo 4 / 5 / 6:',len(A_TABLO4),len(A_TABLO5),len(A_TABLO6))

PREDICTORS=[1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23]
PRED_NO_MISSING=[j for j in PREDICTORS if j not in {9,13}]
assert len(PREDICTORS)==20 and len(PRED_NO_MISSING)==18
B_DATALIST=[];B_DENETIM=[]
for model, fields, scheme in [
    ('Ikili_20_oncul',PREDICTORS,'ikili'),
    ('Besli_20_oncul',PREDICTORS,'besli'),
    ('Ikili_18_oncul_CGPA_kardes_haric',PRED_NO_MISSING,'ikili')
]:
    for sub in ['Depresyon','Anksiyete','Stres']:
        col=sub+('_Kategori' if scheme=='ikili' else '_5_Kategori')
        idx=[i for i in df.index if P.loc[i,col] is not None and all(j in KAYIT_ITEMS[i] for j in fields)]
        N=len(idx)
        baskets=[set(KAYIT_ITEMS[i][j] for j in fields) for i in idx]
        scores=[f'DASS_{sub}_altolcek={P.loc[i,col]}' for i in idx]
        msk=frequency_masks(baskets)
        use,p2,_,_,_=rules_from_masks(msk,N,support_floor=.10,prevalence_cut=.85)
        c_masks=frequency_masks([{x} for x in scores]); full_masks={**use,**c_masks}
        frequent_ante=[(a,) for a in use] + list(p2)
        test_n=len(frequent_ante)*len(c_masks)
        B_DENETIM.append(dict(Model=model,Alt_olcek=sub,N=N,Oncul_degisken=len(fields),
                              Sik_1_oncul=len(use),Sik_2_oncul=len(p2),
                              Yerel_test_sayisi=test_n))
        for A in frequent_ante:
            amask=maskbits(N)
            for x in A: amask &= use[x]
            for c,cmask in c_masks.items():
                both=amask&cmask
                if both.bit_count()/N<.10-1e-12:continue
                met=measure(A,c,both,full_masks,N)
                if met['Guven']<.70-1e-12 or met['Lift']<1.20-1e-12:continue
                B_DATALIST.append(dict(Model=model,Alt_olcek=sub,Oncul=' + '.join(A),Sonuc=c,
                                       Oncul_oge_sayisi=len(A),**met))
B_DENETIM=pd.DataFrame(B_DENETIM)
B_TEST_MAP=B_DENETIM.groupby('Model').Yerel_test_sayisi.sum().to_dict()
B_ALL=pd.DataFrame(B_DATALIST)
if not B_ALL.empty:
    B_ALL['Global_Bonferroni_p']=[min(1.,p*B_TEST_MAP[model]) for model,p in zip(B_ALL.Model,B_ALL.Fisher_p)]
    B_ALL['Uretkenlik_Bonferroni_p']=[min(1.,p*B_TEST_MAP[model]) for model,p in zip(B_ALL.Model,B_ALL.Uretkenlik_p)]
    B_ALL['Ek_katki_duzeltilmis_anlamli']=(B_ALL.Uretkenlik_delta.gt(0)& B_ALL.Uretkenlik_Bonferroni_p.lt(.05))
    B_ALL=B_ALL.sort_values(['Model','Lift','Destek','Oncul'],ascending=[True,False,False,True]).reset_index(drop=True)
B_ANA=main_threshold(B_ALL[B_ALL.Model.eq('Ikili_20_oncul')]) if not B_ALL.empty else B_ALL.copy()
B_DUY=[]
for model in B_TEST_MAP:
    rows=B_ALL[B_ALL.Model.eq(model)] if not B_ALL.empty else B_ALL
    for sup in [.10,.15,.20]:
        for lift in [1.20,1.40,1.60]:
            selected=main_threshold(rows,sup=sup,lift=lift) if not rows.empty else rows
            for sub in ALT_OLCEK:
                r=selected[selected.Alt_olcek.eq(sub)] if not selected.empty else selected
                B_DUY.append(dict(Model=model,Alt_olcek=sub,Destek=sup,Guven=.70,Lift=lift,
                                  Kural=len(r),Global_Bonferroni_genel_p_lt_005=int(r.Global_Bonferroni_p.lt(.05).sum()) if not r.empty else 0,
                                  Ek_katki_anlamli=int(r.Ek_katki_duzeltilmis_anlamli.sum()) if not r.empty else 0))
B_DUY=pd.DataFrame(B_DUY)
print('B kısıtlanmış ana model:',len(B_ANA),'Global test aileleri:',B_TEST_MAP)

C_ROWS=[]
for area,label,filtered in [
 ('DASS21','DASS-21 aynı alt ölçek',A_ANA[A_ANA.Sinif.eq('DASS_ayni_altolcek')]),
 ('DASS21','DASS-21 alt ölçekler arası',A_ANA[A_ANA.Sinif.eq('DASS_altolcekler_arasi')]),
 ('Akademik','Akademik',filter_cross(A_ANA,{'Akademik'})),
 ('Sosyal','Sosyal katılım',filter_cross(A_ANA,{'Sosyal'})),
 ('Aile_ve_yasam','Aile/ikamet',filter_cross(A_ANA,{'Aile_ve_yasam'})),
 ('Demografik','Demografik',filter_cross(A_ANA,{'Demografik'})),
 ('Psikososyal_diger','Uyku/intiharla ilgili özbildirim',A_ANA[A_ANA.Sinif.eq('Psikososyal_alan_ici')]),
]:
    strongest=filtered.sort_values(['Lift','Destek'],ascending=False).head(1)
    example=(strongest.Oncul.iloc[0]+' => '+strongest.Sonuc.iloc[0]) if len(strongest) else 'Ana eşiklerde örnek yok'
    C_ROWS.append({'Gosterge_alani':label,'Ana_esikte_kural':len(filtered),
                   'Global_duzeltilmis_fisher_005':int(filtered.Global_Bonferroni_p.lt(.05).sum()) if not filtered.empty else 0,
                   '2_oncul_duzeltilmis_ek_katki_005':int(filtered[filtered.Oncul_oge_sayisi.eq(2)].Ek_katki_duzeltilmis_anlamli.sum()) if not filtered.empty else 0,
                   'Temsilî_kural':example})
C_OZET=pd.DataFrame(C_ROWS)

BASE_FREQ=pd.DataFrame([{'Oge':item,'Degisken':var(item),'Grup':domain(item),
                         'n':n,'Yuzde_350':round(100*n/len(df),3)}
                        for item,n in Counter(x for d in KAYIT_ITEMS for x in d.values()).items()])
BASE_FREQ=BASE_FREQ.sort_values(['n','Oge'],ascending=[False,True])
EK_DASS=[]
for j in range(1,22):
    col=f'DASS_{j:02d}'
    EK_DASS.append({'Anket_soru_no':32+j,'DASS_madde_no':j,
                    'Alt_olcek':next(s for s,v in ALT_OLCEK.items() if j in v),
                    'Ozgun_ingilizce_soru':df.columns[31+j],
                    **{f'Puan_{p}_n':int((dass_kod[col]==p).sum()) for p in range(4)},
                    'Eksik_n':int(dass_kod[col].isna().sum())})
EK_DASS=pd.DataFrame(EK_DASS)
AUDIT=pd.DataFrame([{'Kontrol':'Ham kayıt','N':len(df_raw)},
                    {'Kontrol':'Dışlanan geçersiz kardeş sayısı kaydı','N':int(hatali.sum())},
                    {'Kontrol':'Temel analiz örneklemi','N':len(df)},
                    {'Kontrol':'A tam vaka örneklemi','N':A_AUDIT['N']},
                    {'Kontrol':'A alternatif tam vaka örneklemi','N':A_SENS_AUDIT['N']},
                    {'Kontrol':'Uyku Satisfied','N':int((df.iloc[:,28].map(txt)=='satisfied').sum())},
                    {'Kontrol':'Uyku Not at all noticeable','N':int((df.iloc[:,30].map(txt)=='not at all noticeable').sum())},
                    {'Kontrol':'Nefes darlığı 0','N':int((dass_kod['DASS_04']==0).sum())},
                    {'Kontrol':'Nefes darlığı eksik','N':int(dass_kod['DASS_04'].isna().sum())}])
assert [int(x) for x in AUDIT.N.tail(4)]==[96,127,193,8]
B_DENETIM=pd.DataFrame(B_DENETIM)
DIAGNOSTIC=pd.DataFrame([A_AUDIT,A_SENS_AUDIT])

OUTPUT=Path('Apriori_Analiz_Ciktilari'); OUTPUT.mkdir(exist_ok=True)
CSV={
 '01_veri_kodlama_denetimi.csv':AUDIT,
 '02_degisken_kategori_frekanslari_350.csv':BASE_FREQ,
 '03_Ek_A1_21_DASS_maddesi.csv':EK_DASS,
 '04_DASS21_guvenirlik.csv':guv_df,
 '05_A_apriori_tum_gevsek_esik_kurallari.csv':A_KURALLAR,
 '06_A_ana_esik_tum_kurallar.csv':A_ANA,
 '07_A_Tablo4_psikososyal.csv':A_TABLO4,
 '08_A_Tablo5_akademik_sosyal.csv':A_TABLO5,
 '09_A_Tablo6_aile_yasam.csv':A_TABLO6,
 '10_A_alanlar_arasi_tum_gevsek_esik.csv':A_CROSS_ALL,
 '23_A_dogrudan_sosyal_aile_oncul_psikososyal_sonuc.csv':A_CROSS_ALL[A_CROSS_ALL.Oncul_alanlari.map(lambda x: not any(g in x.split(';') for g in PSY_GRUP))] if not A_CROSS_ALL.empty else A_CROSS_ALL,
 '11_A_duyarlilik_esikleri.csv':A_DUY,
 '12_A_duyarlilik_CGPA_kardes_haric.csv':A_SENS,
 '13_A_istatistik_aile_ve_N.csv':DIAGNOSTIC,
 '14_B_kisitli_ana_model_kurallari.csv':B_ANA,
 '15_B_kisitli_duyarlilik_tum_kurallar.csv':B_ALL,
 '16_B_duyarlilik_tablosu.csv':B_DUY,
 '17_B_test_aileleri_ve_N.csv':B_DENETIM,
 '21_B_besli_kategori_duyarlilik_kurallari.csv':B_ALL[B_ALL.Model.eq('Besli_20_oncul')] if not B_ALL.empty else B_ALL,
 '22_B_CGPA_kardes_haric_kurallar.csv':B_ALL[B_ALL.Model.eq('Ikili_18_oncul_CGPA_kardes_haric')] if not B_ALL.empty else B_ALL,
 '18_C_sosyolojik_kanıt_ve_yorum_siniri.csv':C_OZET,
 '19_A_tablo_secim_olcutleri.csv':A_TALIMATLAR,
 '20_yorumlanamayan_yanitlar.csv':kod_kayiplari,
}
for fn,table in CSV.items():
    table.to_csv(OUTPUT/fn,index=False,encoding='utf-8-sig')
readme="APRIORI ANALIZ CIKTILARI\nA: 49 degiskenli kural analizi.\nB: 20 onculden DASS-21 alt olceklerine yonelik kisitli analiz ve duyarlilik kontrolleri.\nC: Alan bazinda kural sayilari, duzeltilmis test sonuclari ve ornek kurallar.\nAna esikler: destek >= 0.15; guven >= 0.70; lift >= 1.40.\nOn tarama: destek >= 0.10; guven >= 0.70; lift >= 1.20.\nDuyarlilik: destek 0.10/0.15/0.20; lift 1.20/1.40/1.60.\nYayginligi %85'i asan ogeler kural kesfinden cikarilir.\nFisher sag kuyruk p ve global Bonferroni duzeltmesi uygulanir.\n"
(OUTPUT/'00_OKUYUN.txt').write_text(readme,encoding='utf8')
zip_path=Path('Apriori_Analiz_Ciktilari.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUTPUT.iterdir()):z.write(p,arcname=p.name)
print('TAMAMLANDI. CSV:',len(CSV),' ZIP:',zip_path)
print('A başlangıç N:',A_AUDIT['N'],' | A ana kural:',len(A_ANA),
      '| Tablo 4:',len(A_TABLO4),' | Tablo 5:',len(A_TABLO5),' | Tablo 6:',len(A_TABLO6))
print('B ana kural:',len(B_ANA))
print('A yönlendirilmiş sosyal/akademik/aile öncülü, psikososyal sonuç:', len(CSV['23_A_dogrudan_sosyal_aile_oncul_psikososyal_sonuc.csv']))
display(A_DUY[A_DUY.Model.eq('49_degisken')].head(9))
display(B_DUY.head(9))
try:
    from google.colab import files
    files.download(str(zip_path))
except ImportError:
    pass
