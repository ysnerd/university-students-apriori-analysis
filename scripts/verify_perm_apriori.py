import csv
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('Behavior_Analysis_for_university_ Students (2)(1).xlsx')
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('Apriori_Analiz_Ciktilari')
if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
REPEATS = 499
SEED = 20261120
MIN_S_FREQUENT = .10
MIN_S_RULE = .15
MIN_CONF = .70
MIN_LIFT = 1.40
PREVALENCE_MAX = .85

AD = {
 1:'Yaş',4:'Cinsiyet',5:'Din',6:'Daimi_ikamet',7:'Mevcut_ikamet',8:'Aile_tipi',
 9:'Kardeş_sayısı',10:'Baba_mesleği',11:'Anne_mesleği',12:'Aile_ortamı',
 13:'CGPA',14:'Günlük_çalışma',15:'Sosyal_medya',16:'İlişki_durumu',
 17:'Sigara',18:'Dini_uygulama',19:'Akademik_memnuniyet',20:'Ders_dışı_katılım',
 22:'Kariyer_fırsatı',23:'Mesleğin_sosyal_değeri',24:'İntihar_madde_25',
 25:'İntihar_madde_26',26:'İntihar_madde_27',27:'İntihar_madde_28',
 28:'Uyku_memnuniyeti',29:'Uyku_işlev',30:'Uyku_farkedilme',31:'Uyku_endişe',
}
UYKU_MEMNUN={'very satisfied':'Çok_memnun','satisfied':'Memnun','moderately satisfied':'Orta','dissatisfied':'Memnuniyetsiz','very dissatisfied':'Çok_memnuniyetsiz'}
UYKU_FARK={'not at all noticeable':'Hiç_fark_edilmez','barely':'Zor_fark_edilir','somewhat':'Kısmen','much':'Fazla','very much noticeable':'Çok_fazla'}
UYKU_ISLEV={'not at all interfering':'Hiç_etkilemiyor','a little':'Biraz','somewhat':'Kısmen','much':'Fazla','very much interfering':'Çok_fazla'}
UYKU_ENDISE={'not at all':'Hiç','barely':'Çok_az','somewhat':'Kısmen','much':'Fazla','very much':'Çok_fazla'}
KATEGORI_HARITA={28:UYKU_MEMNUN,29:UYKU_ISLEV,30:UYKU_FARK,31:UYKU_ENDISE}
DASS_CEVAP={'did not apply to me at all':0,'applied to me to some degree, or some of the time':1,'applied to me to a considerable degree or a good part of time':2,'applied to me very much or most of the time':3}
KARDES_IFADE={'1 big sister':1,'1 elder sister':1,'1 sister.':1,'1 brother 1 sister':2,'2 sister':2,'3 brothers':3,'3 sisters':3,'numbers: 1':1}
ALT_OLCEK={'Depresyon':[3,5,10,13,16,17,21],'Anksiyete':[2,4,7,9,15,19,20],'Stres':[1,6,8,11,12,14,18]}
GRUPLAR={
 'Akademik': {'CGPA','Günlük_çalışma','Akademik_memnuniyet','Kariyer_fırsatı','Mesleğin_sosyal_değeri'},
 'Sosyal': {'Sosyal_medya','Ders_dışı_katılım','İlişki_durumu','Sigara','Dini_uygulama'},
 'Aile_ve_yasam': {'Daimi_ikamet','Mevcut_ikamet','Aile_tipi','Kardeş_sayısı','Baba_mesleği','Anne_mesleği','Aile_ortamı'},
 'Demografik': {'Yaş','Cinsiyet','Din'},
 'Psikososyal_diger': {'İntihar_madde_25','İntihar_madde_26','İntihar_madde_27','İntihar_madde_28','Uyku_memnuniyeti','Uyku_işlev','Uyku_farkedilme','Uyku_endişe'},
 'DASS21_madde': {f'DASS_{j:02d}' for j in range(1,22)},
}
GRUP_MAP={x:k for k,s in GRUPLAR.items() for x in s}
DASS_TO_SUB={f'DASS_{j:02d}':sub for sub,items in ALT_OLCEK.items() for j in items}
PSY_GRUP={'DASS21_madde','Psikososyal_diger'}
OTHER_GRUP={'Akademik','Sosyal','Aile_ve_yasam','Demografik'}
GROUPS=['DASS_ayni_altolcek','DASS_altolcekler_arasi','Psikososyal_alan_ici','Alanlar_arasi','Psikososyal_olmayan']

def txt(v):
    if v is None: return ''
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(v))).strip().casefold()

def eksik(v): return txt(v) in {'','n/a','na','null','none','-','nan'}

def sibling(v):
    s=txt(v)
    if s in {'none','no','no one'}: return 0
    if s in KARDES_IFADE: return KARDES_IFADE[s]
    if s == '10+': return 10
    if re.fullmatch(r'0*\d{1,3}',s):
        n=int(s)
        return n if n<=30 else None
    return None

def cgpa(v):
    s=txt(v).replace(',','.')
    m=re.fullmatch(r'(?:cgpa\s*[-:]?\s*)?(\d(?:\.\d{1,2})?)',s)
    if not m: return None
    x=float(m.group(1))
    if not 0<x<=4: return None
    return '0-2.00' if x<=2 else ('2.01-3.00' if x<=3 else ('3.01-3.50' if x<=3.5 else '3.51-4.00'))

def age(v):
    s=txt(v)
    if s in {'19 years 11 months 24 days','20 years 9 months 30 days'}: return '20_ve_altı'
    if s=='22.5': return '21-22'
    if s in {'24+','years: 25'}: return '23_ve_üstü'
    if not re.fullmatch(r'\d{2}',s): return None
    n=int(s)
    if not 16<=n<=80: return None
    return '20_ve_altı' if n<=20 else ('21-22' if n<=22 else '23_ve_üstü')

def encode_non_dass(i,v):
    if eksik(v) and not(i==9 and txt(v)=='none'): return None
    s=txt(v)
    if i==1: cat=age(v)
    elif i==9:
        n=sibling(v)
        cat=None if n is None else ('0' if n==0 else ('1' if n==1 else ('2' if n==2 else '3_ve_üstü')))
    elif i==13: cat=cgpa(v)
    elif i in KATEGORI_HARITA:
        cat=KATEGORI_HARITA[i].get(s)
        if cat is None: raise ValueError(f'Unexpected sleep response {i}: {v!r}')
    else:
        cat={'yes':'Evet','no':'Hayır','neural':'Neutral','offen':'Often',
             'strongly_disagree':'Strongly-disagree','dis-agree':'Disagree',
             'join or extended':'Joint_or_Extended'}.get(s,re.sub(r'\s+','_',s))
    return None if cat is None else f'{AD[i]}={cat}'

def load_codes():
    raw_table=pd.read_excel(INPUT,sheet_name=0,dtype=object,keep_default_na=False)
    rows=[list(raw_table.columns)]+raw_table.values.tolist()
    if len(rows)!=352 or len(rows[0])!=53 or not txt(rows[0][0]).startswith('semester('):
        raise ValueError(f'Unexpected spreadsheet layout: {len(rows)} rows')
    original=rows[1:]
    bad=[i for i,r in enumerate(original) if txt(r[9]).lstrip('0').isdigit() and int(txt(r[9]))>1000]
    assert len(bad)==1, f'Unexpected excluded rows: {bad}'
    cleaned=[r for i,r in enumerate(original) if i not in bad]
    assert len(cleaned)==350
    keys=list(AD)+list(range(32,53))
    code_rows=[]
    for r in cleaned:
        record={}
        for j in AD:
            x=encode_non_dass(j,r[j])
            if x is not None: record[j]=x
        for j in range(1,22):
            v=r[31+j]
            code=None if eksik(v) else DASS_CEVAP.get(txt(v))
            if code is None and not eksik(v):raise ValueError(f'Unexpected DASS response: {v!r}')
            if code is not None:record[31+j]=f'DASS_{j:02d}={code}'
        code_rows.append(record)
    clean=[r for r in code_rows if len(r)==49]
    assert len(clean)==328, f'Unexpected complete cases: {len(clean)}'
    names=[AD[i] if i in AD else f'DASS_{i-31:02d}' for i in keys]
    assert len(set(names))==49
    rows2=[[record[i] for i in keys] for record in clean]
    code_names=[]
    cols=[]
    cat_info=[]
    min_single=math.ceil(.10*len(rows2)-1e-10)
    allcounts=Counter(x for row in rows2 for x in row)
    for j,col_name in enumerate(names):
        cats=sorted({r[j] for r in rows2})
        cols.append({c:i for i,c in enumerate(cats)})
        for c in cats:
            n=allcounts[c]
            if min_single<=n<=.85*len(rows2):
                code_names.append(c)
                cat_info.append((j,cols[-1][c],c,col_name))
    coded=np.asarray([[cols[j][r[j]] for j in range(49)] for r in rows2],dtype=np.int16)
    assert len(code_names)==119, f'Expected 119 categories, got {len(code_names)}'
    return coded,cat_info,code_names

def masks_from_codes(coded,cat_info):
    return [int.from_bytes(np.packbits(coded[:,j]==k,bitorder='little').tobytes(),'little')
            for j,k,_,_ in cat_info]

def get_class(items, cat_info):
    variables={cat_info[i][3] for i in items}
    domains={GRUP_MAP[x] for x in variables}
    if domains=={'DASS21_madde'}:
        subs={DASS_TO_SUB[x] for x in variables}
        return 'DASS_ayni_altolcek' if len(subs)==1 else 'DASS_altolcekler_arasi'
    if domains&PSY_GRUP and domains&OTHER_GRUP: return 'Alanlar_arasi'
    if domains&PSY_GRUP: return 'Psikososyal_alan_ici'
    return 'Psikososyal_olmayan'

def evaluate(masks,cat_info, N, collect_candidates=False):
    needs10=math.ceil(.10*N-1e-10)
    needs15=math.ceil(.15*N-1e-10)
    eligible=[i for i,m in enumerate(masks) if needs10<=m.bit_count()<=.85*N]
    singles={i:masks[i].bit_count() for i in eligible}
    pair10=[]
    pair15={}
    cand=Counter()
    cand_support=Counter()
    rules=Counter()
    for a,b in combinations(eligible,2):
        if cat_info[a][0]==cat_info[b][0]: continue
        n=(masks[a]&masks[b]).bit_count()
        if n>=needs10:
            pair10.append((a,b))
            if n>=needs15:pair15[(a,b)]=n
    if collect_candidates:
        for i in eligible:
            for c in eligible:
                if cat_info[i][0]!=cat_info[c][0]:cand[get_class((i,c),cat_info)]+=1
        for a,b in pair10:
            for c in eligible:
                if cat_info[c][0] not in (cat_info[a][0],cat_info[b][0]):
                    cand[get_class((a,b,c),cat_info)]+=1
    for (a,b),n in pair15.items():
        cat=get_class((a,b),cat_info)
        cand_support[cat]+=2
        for prev, cons in ((b,a),(a,b)):
            conf=n/singles[prev]
            lift=conf/(singles[cons]/N)
            if conf>=MIN_CONF-1e-12 and lift>=MIN_LIFT-1e-12:
                rules[cat]+=1
    neighbours=defaultdict(set)
    for a,b in pair15: neighbours[a].add(b)
    for (a,b),ab_count in pair15.items():
        possible=neighbours[a].intersection(neighbours[b])
        for c in possible:
            if c<=b: continue
            if cat_info[c][0] in {cat_info[a][0],cat_info[b][0]}:continue
            mask=masks[a]&masks[b]&masks[c]
            count=mask.bit_count()
            if count<needs15:continue
            cat=get_class((a,b,c),cat_info)
            cand_support[cat]+=3
            for x,y,z in ((a,b,c),(a,c,b),(b,c,a)):
                an=pair15[tuple(sorted((x,y)))]
                conf=count/an
                lift=conf/(singles[z]/N)
                if conf>=MIN_CONF-1e-12 and lift>=MIN_LIFT-1e-12:
                    rules[cat]+=1
    return rules,cand,cand_support,len(eligible),len(pair10),len(pair15)

def write_csv(path,header,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f,delimiter=';')
        w.writerow(header)
        w.writerows(rows)

def main():
    coded,meta,names=load_codes()
    N=coded.shape[0]
    assert N==328
    observed,candidates,pass_support,ncat,npair10,npair15=evaluate(masks_from_codes(coded,meta),meta,N,True)
    print('Observed',dict(observed),'total',sum(observed.values()),'candidates',sum(candidates.values()),'ncat',ncat,'pairs10',npair10,'pairs15',npair15,flush=True)
    EXPECTED=[293,3232,1259,2246,6]
    assert [observed[k] for k in GROUPS]==EXPECTED, 'Observed rules do not reproduce the published 7,036!'
    assert sum(candidates.values())==360625, 'Candidate family differs from previous reported number'
    assert sum(pass_support.values())==34265, 'Support-qualified candidates differ'
    print('Original 7,036 and 360,625 hypothesis candidates reproduced.',flush=True)

    rng=np.random.default_rng(SEED)
    results=[]
    for i in range(1,REPEATS+1):
        randomized=np.empty_like(coded)
        for j in range(coded.shape[1]):
            randomized[:,j]=coded[rng.permutation(N),j]
        rules,_,_,_,_,_=evaluate(masks_from_codes(randomized,meta),meta,N)
        results.append([rules[g] for g in GROUPS])
        if i%100==0 or i==REPEATS:
            print('Completed permutations:',i,'/',REPEATS,flush=True)
    arr=np.asarray(results,dtype=int)
    rows=[]
    for j,group in enumerate(GROUPS):
        data=arr[:,j]
        rows.append([group,int(observed[group]),REPEATS,round(float(data.mean()),4),int(data.min()),int(data.max()),int((data>=observed[group]).sum()),round(float((1+(data>=observed[group]).sum())/(REPEATS+1)),4)])
    totals=arr.sum(axis=1)
    rows.append(['TOPLAM',sum(observed.values()),REPEATS,round(float(totals.mean()),4),int(totals.min()),int(totals.max()),int((totals>=sum(observed.values())).sum()),round(float((1+(totals>=sum(observed.values())).sum())/(REPEATS+1)),4)])
    write_csv(OUT/'Kural_Permutasyon_Yeniden_Dogrulama_Ozeti.csv',['Kural grubu','Gozlenen','Tekrar','Bagimsizlik_nulu_ortalama','En_dusuk','En_yuksek','Gozlenenden_buyuk_esit','Monte_Carlo_p'],rows)
    write_csv(OUT/'Kural_Permutasyon_499_Ham_Tekrarlar.csv',['Tekrar',*GROUPS,'TOPLAM'],[[i+1,*r,sum(r)] for i,r in enumerate(arr.tolist())])
    write_csv(OUT/'Kural_Olanaklari_Yeniden_Dogrulama.csv',['Kural grubu','Uygun aday','Destek_015_saglayan','Tüm_esikler','Adayda_oran_yuzde'],[[g,candidates[g],pass_support[g],observed[g],round(observed[g]/candidates[g]*100,4)] for g in GROUPS])
    (OUT/'Rastgelelestirme_Yontemi_ve_Sinirlar.txt').write_text(f'''İncelenen veri: {INPUT.name}. Ham 351 kayıt, kardeş sayısı geçersiz 1 kayıt çıkarılır; 350 temiz kayıt; 49 değişkende tam vaka n=328.\n49 değişken bağımsız olarak katılımcılar arasında yeniden karıştırıldı; her değişkenin tüm kategori frekansları sabit kaldı.\nHer tekrar 119 uygun kategoride Apriori benzeri ikili ve üçlü öğe birleşimlerini taradı; alt yaygınlık 0,10, üst 0,85; bir veya iki öncüllü, tek sonuçlu kurallar.\nKural eşikleri destek >= 0,15, güven >= 0,70, lift >= 1,40. Üç öğe en fazla; aynı değişkenin iki farklı kategorisi aynı kuralda bulunamaz.\nTekrar={REPEATS}; sabit NumPy tohum={SEED}.\nGözlenen kural: {sum(observed.values())}; rastgeleleştirmede ortalama {totals.mean():.4f}, maks {totals.max()}, gözlenen kadar veya üzeri {int((totals>=sum(observed.values())).sum())}; +1 düzeltmeli Monte Carlo p={(1+(totals>=sum(observed.values())).sum())/(REPEATS+1):.4f}.\nKarşılaştırma null modeli değişkenler arasında tam bağımsızlıktır. Bu karşılaştırma DASS-21 faktör yapısını kanıtlamaz ve bağımsız örneklemde kararlılığı sınamaz.\nBu analizde bağımsız dağılım altında yüksek liftli kuralların çok az çıkması özellikle psikolojik maddeler arasındaki bağımlılığın bilinen özelliğini de yansıtır.\nKural alanları karşılıklı dışlayıcı sınıflandırılmıştır, kurallara hangi alana ait olduklarına göre ayrı p değerleri atanmamıştır; çoklu karşılaştırma / seçilim sorunları ayrıca ele alınmalıdır.\n''',encoding='utf8')
    for row in rows:print(row,flush=True)

if __name__=='__main__':main()
