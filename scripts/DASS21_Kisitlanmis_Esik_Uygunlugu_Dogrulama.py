from pathlib import Path
import argparse
import pandas as pd
import re, math,unicodedata, collections,itertools
argp=argparse.ArgumentParser(description="Restricted DASS21 lift threshold feasibility")
argp.add_argument('--excel', type=Path, required=True)
argp.add_argument('--output', type=Path, default=Path('DASS21_Kisitlanmis_Esik_Uygunlugu.csv'))
args=argp.parse_args()
p=str(args.excel.expanduser().resolve())
if not Path(p).is_file(): raise FileNotFoundError(p)
raw_table=pd.read_excel(p,sheet_name=0,dtype=object,keep_default_na=False)
r=[list(raw_table.columns)]+raw_table.values.tolist()
rows=r[1:]
def txt(value):
 if value is None: return ''
 value=unicodedata.normalize('NFKC',str(value))
 return re.sub(r'\s+',' ',value).strip().casefold()
EKSIK={'','n/a','na','null','none','-','nan'}
def eksik(v):return txt(v) in EKSIK
KARDES_IFADE={'1 big sister':1,'1 elder sister':1,'1 sister.':1,'1 brother 1 sister':2,'2 sister':2,'3 brothers':3,'3 sisters':3,'numbers: 1':1}
def kardeş_sayisi(v):
 s=txt(v)
 if s in {'none','no','no one'}:return 0
 if s in KARDES_IFADE:return KARDES_IFADE[s]
 if s=='10+':return 10
 if re.fullmatch(r'0*\d{1,3}',s):
  num=int(s);return num if num<=30 else None
 return None
def cgpa_kategori(value):
 s=txt(value).replace(',','.')
 m=re.fullmatch(r'(?:cgpa\s*[-:]?\s*)?(\d(?:\.\d{1,2})?)',s)
 if not m:return None
 x=float(m.group(1))
 if not (0<x<=4):return None
 return '0-2.00' if x<=2 else ('2.01-3.00' if x<=3 else ('3.01-3.50' if x<=3.5 else '3.51-4.00'))
def yas_araligi(value):
 s=txt(value)
 if s in {'19 years 11 months 24 days','20 years 9 months 30 days'}:return '20_ve_altı'
 if s=='22.5':return '21-22'
 if s in {'24+','years: 25'}:return '23_ve_üstü'
 if not re.fullmatch(r'\d{2}',s):return None
 n=int(s)
 if not 16<=n<=80:return None
 return '20_ve_altı' if n<=20 else ('21-22' if n<=22 else '23_ve_üstü')
AD={1:'Yaş',4:'Cinsiyet',5:'Din',6:'Daimi_ikamet',7:'Mevcut_ikamet',8:'Aile_tipi',9:'Kardeş_sayısı',10:'Baba_mesleği',11:'Anne_mesleği',12:'Aile_ortamı',13:'CGPA',14:'Günlük_çalışma',15:'Sosyal_medya',16:'İlişki_durumu',17:'Sigara',18:'Dini_uygulama',19:'Akademik_memnuniyet',20:'Ders_dışı_katılım',22:'Kariyer_fırsatı',23:'Mesleğin_sosyal_değeri'}
def deger_kodla(i,v):
 if eksik(v) and not (i==9 and txt(v)=='none'): return None
 s=txt(v)
 if i==1: cat=yas_araligi(v)
 elif i==9:
  n=kardeş_sayisi(v)
  if n is None:return None
  cat='0' if n==0 else ('1' if n==1 else ('2' if n==2 else '3_ve_üstü'))
 elif i==13:cat=cgpa_kategori(v)
 else:
  cat={'yes':'Evet','no':'Hayır','neural':'Neutral','offen':'Often','strongly_disagree':'Strongly-disagree','dis-agree':'Disagree','join or extended':'Joint_or_Extended'}.get(s,re.sub(r'\s+','_',s))
 if cat is None:return None
 return f'{AD[i]}={cat}'
DASS_CEVAP={'did not apply to me at all':0,'applied to me to some degree, or some of the time':1,'applied to me to a considerable degree or a good part of time':2,'applied to me very much or most of the time':3}
ALT={'Depresyon':[3,5,10,13,16,17,21],'Anksiyete':[2,4,7,9,15,19,20],'Stres':[1,6,8,11,12,14,18]}
NORMAL={'Depresyon':9,'Anksiyete':7,'Stres':14}
fields= list(AD)
valid_rows=[];removed=[]
for z,row in enumerate(rows):
 s=txt(row[9]);bad=s.lstrip('0').isdigit() and int(s)>1000
 if bad:removed.append(z+2);continue
 vals={i:deger_kodla(i,row[i]) for i in fields}
 codes=[DASS_CEVAP.get(txt(row[32+j]),None) for j in range(21)]
 valid_rows.append((z+2,vals,codes))
print('original',len(rows),'clean',len(valid_rows),'removed record count',len(removed))
results=[]
for sub,items in ALT.items():
 eligible=[]
 for z,vals,codes in valid_rows:
  if not all(vals.values()):continue
  selected=[codes[j-1] for j in items]
  if any(x is None for x in selected):continue
  raw=sum(selected);scaled=2*raw
  outcome='Referans_üstü' if scaled>NORMAL[sub] else 'Referans_aralığı'
  eligible.append((z,vals,outcome))
 N=len(eligible)
 prev=collections.Counter(e[2] for e in eligible)
 masks=collections.defaultdict(int)
 for idx,(_,vals,_) in enumerate(eligible):
  for item in vals.values(): masks[item]|=1<<idx

 need=math.ceil(.10*N-1e-10)
 use={k:v for k,v in masks.items() if need<=v.bit_count() and v.bit_count()/N<=.85}
 ordered=sorted(use)
 frequent_pair=[]
 for a,b in itertools.combinations(ordered,2):
  if a.split('=')[0]==b.split('=')[0]:continue
  if (use[a]&use[b]).bit_count()>=need: frequent_pair.append((a,b))
 candidates=[(a,) for a in use]+frequent_pair
 min_support=math.ceil(.15*N-1e-10)
 print('\n',sub,'N=',N,'categories',dict(prev),'single=',len(use),'pair=',len(frequent_pair),'candidates=',len(candidates))
 for cat,nC in prev.items():
  pC=nC/N; maxlift=1/pC
  reqconf=max(.70,1.4*pC)
  possible=0; possible_simple=0
  for A in candidates:
   mm=(1<<N)-1
   for a in A:mm &= use[a]
   nA=mm.bit_count()
   if nA>=min_support and nC>=min_support:
    possible_simple+=1
    if min(nA,nC)/nA+1e-12>=reqconf:
     possible+=1
  print(cat,'nC',nC,'prevalence',pC,'max lift',maxlift,'required confidence',reqconf,'mathematically possible antecedents',possible,'base supp feasible',possible_simple)
  results.append((sub,cat,N,nC,pC,maxlift,reqconf,len(candidates),possible,possible_simple))
import csv
out=str(args.output.expanduser().resolve())
Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out,'w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['Alt olcek','Sonuc kategori','N','n','Prevalans','Maksimum lift','Gerekli asgari guven','Aday oncul sayisi','Esikleri matematiksel olarak saglayabilecek oncul sayisi','Destek icin uygun oncul sayisi']);w.writerows(results)
print('SAVED',out)
