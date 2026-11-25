from __future__ import annotations
import csv, math, re, unicodedata
from collections import Counter
from pathlib import Path
import argparse
import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np

argp=argparse.ArgumentParser(description="DASS21 uncategorized-score sensitivity analysis (12 tests)")
argp.add_argument('--excel',type=Path,required=True,help='Original XLSX dataset')
argp.add_argument('--output',type=Path,default=Path('DASS21_Surekli_Puan_Duyarlilik_12_Karsilastirma.csv'))
args=argp.parse_args()
INPUT=args.excel.expanduser().resolve()
OUTPUT=args.output.expanduser().resolve()
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
if not INPUT.is_file(): raise FileNotFoundError(INPUT)

raw_table=pd.read_excel(INPUT,sheet_name=0,dtype=object,keep_default_na=False)
all_rows=[list(raw_table.columns)]+raw_table.values.tolist()
assert len(all_rows)==352 and len(all_rows[0])==53
rows=all_rows[1:]
assert len(rows)==351

def norm(v):
    if v is None: return ''
    s=unicodedata.normalize('NFKC',str(v))
    return re.sub(r'\s+',' ',s).strip().casefold()

EKSIK={'','n/a','na','null','none','-','nan'}
ans={
 'did not apply to me at all':0,
 'applied to me to some degree, or some of the time':1,
 'applied to me to a considerable degree or a good part of time':2,
 'applied to me very much or most of the time':3,
}

KARDES_IFADE = {
 '1 big sister':1,'1 elder sister':1,'1 sister.':1,
 '1 brother 1 sister':2,'2 sister':2,
 '3 brothers':3,'3 sisters':3,'numbers: 1':1,
}

def sibling(v):
    s=norm(v)
    if s in {'none','no','no one'}:return 0
    if s in KARDES_IFADE:return KARDES_IFADE[s]
    if s=='10+':return 10
    if re.fullmatch(r'0*\d{1,3}',s):
        n=int(s)
        return n if n<=30 else None
    return None

def age(v):
    s=norm(v)
    if s in {'19 years 11 months 24 days','20 years 9 months 30 days','22.5','24+','years: 25'}:
        return True
    return bool(re.fullmatch(r'\d{2}',s) and 16<=int(s)<=80)

def cgpa(v):
    s=norm(v).replace(',','.')
    m=re.fullmatch(r'(?:cgpa\s*[-:]?\s*)?(\d(?:\.\d{1,2})?)',s)
    return bool(m and 0<float(m.group(1))<=4)

sleep_valid={
    28:{'very satisfied','satisfied','moderately satisfied','dissatisfied','very dissatisfied'},
    29:{'not at all interfering','a little','somewhat','much','very much interfering'},
    30:{'not at all noticeable','barely','somewhat','much','very much noticeable'},
    31:{'not at all','barely','somewhat','much','very much'},
}

AD=[1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23,24,25,26,27,28,29,30,31]
assert len(AD)==28

def valid_cell(row,j):
    v=row[j]; s=norm(v)
    if j==9:return sibling(v) is not None
    if s in EKSIK:return False
    if j==1:return age(v)
    if j==13:return cgpa(v)
    if j in sleep_valid:return s in sleep_valid[j]
    return True

invalid=[r for r in rows if norm(r[9]).lstrip('0').isdigit() and int(norm(r[9]))>1000]
assert len(invalid)==1
clean=[r for r in rows if r not in invalid]
assert len(clean)==350
complete=[]
for r in clean:
    valid=all(valid_cell(r,j) for j in AD) and all(norm(r[32+i]) in ans for i in range(21))
    if valid: complete.append(r)
assert len(complete)==328, f'Beklenen N=328, elde edilen N={len(complete)}'

subscales={
 'Depresyon':[3,5,10,13,16,17,21],
 'Anksiyete':[2,4,7,9,15,19,20],
 'Stres':[1,6,8,11,12,14,18],
}

def score(row,items):
    return 2*sum(ans[norm(row[31+j])] for j in items)

checks=[
 ('Günlük ders çalışma süresi',14,'4 saat veya üzeri','1-3 saat',lambda s:s in {'4-6 hours','7-9 hours','>=9 hours'},lambda s:s=='1-3 hours'),
 ('Günlük sosyal medya kullanımı',15,'4 saat veya üzeri','1-3 saat',lambda s:s in {'4-6 hours','7-9 hours','>=9 hours'},lambda s:s=='1-3 hours'),
 ('Akademik faaliyetlerden memnuniyet',19,'Evet','Hayır',lambda s:s=='yes',lambda s:s=='no'),
 ('Ders dışı etkinliğe katılım',20,'Evet','Hayır',lambda s:s=='yes',lambda s:s=='no'),
]

def fmt(x,nd=5):return round(float(x),nd)

records=[]
for indicator,col,high_label,low_label,high_pred,low_pred in checks:
    for sub,items in subscales.items():
        hi=[];lo=[]
        for r in complete:
            v=norm(r[col]);s=score(r,items)
            if high_pred(v):hi.append(s)
            elif low_pred(v):lo.append(s)
            else:raise ValueError(f'Beklenmeyen yanıt: {indicator}={v}')
        n1,n0=len(hi),len(lo)
        assert n1+n0==328
        res=mannwhitneyu(hi,lo,alternative='two-sided',method='asymptotic')
        U=float(res.statistic)
        delta=2*U/(n1*n0)-1
        rec={
          'gosterge':indicator,'alt_olcek':sub,
          'grup_1':high_label,'n_1':n1,
          'medyan_1':fmt(np.median(hi),3),
          'q1_1':fmt(np.quantile(hi,.25),3),'q3_1':fmt(np.quantile(hi,.75),3),
          'grup_0':low_label,'n_0':n0,
          'medyan_0':fmt(np.median(lo),3),
          'q1_0':fmt(np.quantile(lo,.25),3),'q3_0':fmt(np.quantile(lo,.75),3),
          'U_grup1':fmt(U,3),
          'rank_biserial':fmt(delta),
          'ham_p':fmt(res.pvalue,9),
          'bonferroni_p_12':fmt(min(res.pvalue*12,1.),9),
          'orneklem_N':328
        }
        records.append(rec)
assert len(records)==12
with OUTPUT.open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
print('Tam vaka N=',len(complete))
print('DASS-21 toplam skor aralıkları:')
for sub,items in subscales.items():
    v=[score(r,items) for r in complete]
    print(sub, 'n',len(v),'median',np.median(v),'IQR',np.quantile(v,[.25,.75]),'minmax',min(v),max(v))
print('\nGösterge\tAlt ölçek\tn1/n0\tmedyanlar\tdelta\tham p\tbonferroni p')
for r in records:
    print(f"{r['gosterge']}\t{r['alt_olcek']}\t{r['n_1']}/{r['n_0']}\t{r['medyan_1']}/{r['medyan_0']}\t{r['rank_biserial']:+.3f}\t{r['ham_p']:.6f}\t{r['bonferroni_p_12']:.6f}")
print('\nSaved',OUTPUT)
