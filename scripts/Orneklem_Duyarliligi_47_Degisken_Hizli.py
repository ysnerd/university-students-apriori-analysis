from __future__ import annotations
import argparse,csv,json,os
from pathlib import Path

import pandas as pd

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--excel',type=Path,required=True)
ap.add_argument('--notebook',type=Path,required=True)
ap.add_argument('--output-dir',type=Path,required=True)
ap.add_argument('--primary-rules',type=Path,required=True)
args=ap.parse_args()
args.output_dir.mkdir(parents=True,exist_ok=True)
os.environ['APR_EXCEL']=str(args.excel.resolve())
nt=json.loads(args.notebook.read_text(encoding='utf-8'))
source='\n'.join(''.join(c['source']) for c in nt['cells'] if c['cell_type']=='code')
marker='\nA_KURALLAR,A_AUDIT=build_exploratory('
assert marker in source
ns={'__name__':'__main__'}
exec(compile(source.split(marker,1)[0],str(args.notebook),'exec'),ns)
fields47=[i for i in ns['FIELDS_ALL'] if i not in (9,13)]
assert len(fields47)==47
orig_df=ns['df'].copy()
orig_items=ns['KAYIT_ITEMS']
complete=[i for i in orig_df.index if all(j in orig_items[i] for j in ns['FIELDS_ALL'])]
assert len(complete)==328

ns['df']=orig_df.loc[complete].copy()
def model_context():
    frames=[set(ns['KAYIT_ITEMS'][i][j] for j in fields47) for i in ns['df'].index
            if all(j in ns['KAYIT_ITEMS'][i] for j in fields47)]
    N=len(frames)
    bits=ns['frequency_masks'](frames)
    used,p2,p3,C,m=ns['rules_from_masks'](bits,N,support_floor=.10,prevalence_cut=.85)
    return N,used,p2,p3,m
n328,used328,p2_328,p3_328,m328=model_context()
assert n328==328
base=pd.read_csv(args.primary_rules,encoding='utf-8-sig')
assert len(base)==7036
eligible=~(base['Oncul']+' + '+base['Sonuc']).str.contains('CGPA=|Kardeş_sayısı=',regex=True)
result328=base.loc[eligible].copy()
assert len(result328)==6973,len(result328)
result328['Model']='47_degisken_ayni_328_katilimci'
result328['Global_Bonferroni_p']=(result328['Fisher_p']*m328).clip(upper=1)
result328['Uretkenlik_Bonferroni_p']=(result328['Uretkenlik_p']*m328).clip(upper=1)
result328['Ek_katki_duzeltilmis_anlamli']=((result328['Uretkenlik_delta']>0) &
                                           (result328['Uretkenlik_Bonferroni_p']<.05))
path328=args.output_dir/'Orneklem_47_Degisken_Ayni_N328_6973_Kural.csv'
result328.to_csv(path328,index=False,encoding='utf-8-sig')
print('Same-sample 47 variables N=328, rules=',len(result328),'candidate family=',m328,flush=True)

invalid=ns['df_raw'].loc[ns['hatali']]
assert len(invalid)==1
r=invalid.iloc[0]
values={}
for j in ns['AD']:
    v=ns['deger_kodla'](j,r.iloc[j])
    if v is not None:values[j]=v
for j in range(1,22):
    v=ns['DASS_CEVAP'].get(ns['txt'](r.iloc[31+j]))
    if v is not None:values[31+j]=f'DASS_{j:02d}={v}'
assert len(values)==48 and 9 not in values
ns['KAYIT_ITEMS']=orig_items+[values]
ns['df']=pd.concat([orig_df,invalid],ignore_index=True)
N,used,p2,p3,m=model_context()
assert N==343,N

def create_rule(A,c,inter):
    nAC=inter.bit_count()
    if nAC/N<.15-1e-12:return None
    mask_a=ns['maskbits'](N)
    for key in A:mask_a &= used[key]
    nA=mask_a.bit_count();nC=used[c].bit_count()
    if not nA or not nC:return None
    conf=nAC/nA;lift=conf/(nC/N)
    if conf<.70-1e-12 or lift<1.40-1e-12:return None
    met=ns['measure'](A,c,inter,used,N)
    row=dict(Model='47_degisken_dislanan_kayit_geri_alindi_343',
             Sinif=ns['class_rule'](A,c),Oncul=' + '.join(A),Sonuc=c,
             Oncul_oge_sayisi=len(A),
             Oncul_alanlari=';'.join(sorted({ns['domain'](x) for x in A})),
             Sonuc_alani=ns['domain'](c),**met)
    row['Global_Bonferroni_p']=min(1.,met['Fisher_p']*m)
    row['Uretkenlik_Bonferroni_p']=min(1.,met['Uretkenlik_p']*m)
    row['Uretkenlik_pozitif']=bool(met['Uretkenlik_delta']>0)
    row['Ek_katki_duzeltilmis_anlamli']=bool(met['Uretkenlik_delta']>0 and row['Uretkenlik_Bonferroni_p']<.05)
    row['Cevap_0_icerir']=any(ns['domain'](x)=='DASS21_madde' and x.endswith('=0') for x in list(A)+[c])
    return row

out=[]
for (a,b),inter in p2.items():
    for A,c in [((a,),b),((b,),a)]:
        val=create_rule(A,c,inter)
        if val is not None:out.append(val)
for (a,b,c),inter in p3.items():
    for A,z in [((a,b),c),((a,c),b),((b,c),a)]:
        val=create_rule(A,z,inter)
        if val is not None:out.append(val)
result343=pd.DataFrame(out,columns=ns['FULL_COLUMNS'])
result343=result343.sort_values(['Lift','Destek','Oncul','Sonuc'],ascending=[False,False,True,True]).reset_index(drop=True)
assert len(result343)==7120,(len(result343),m)
path343=args.output_dir/'Orneklem_47_Degisken_Kayit_Dahil_N343_7120_Kural.csv'
result343.to_csv(path343,index=False,encoding='utf-8-sig')
summary=args.output_dir/'Orneklem_Duyarliligi_328_343_Ozet.csv'
with summary.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f)
    w.writerow(['Analiz','N','Degisken','Kural_sayisi','Dosya'])
    w.writerow(['47 değişken; ana analizle aynı katılımcılar',328,47,len(result328),path328.name])
    w.writerow(['47 değişken; kardeş sayısı geçersiz kayıt geri alındı',343,47,len(result343),path343.name])
print('Restored-sample 47 variables N=343, rules=',len(result343),'candidate family=',m,flush=True)
print('Output:',path328,path343,summary)
