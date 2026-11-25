from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import verify_perm_apriori as v

parser=argparse.ArgumentParser(description='Bootstrap of fixed 30 representative rules')
parser.add_argument('--excel',type=Path,required=True,help='Original XLSX dataset')
parser.add_argument('--selected-dir',type=Path,default=Path(__file__).resolve().parent.parent/'Çıktılar')
parser.add_argument('--output',type=Path,default=Path('Tablo_B2_30_Kural_Bootstrap_Yeniden.csv'))
parser.add_argument('--repeats',type=int,default=3000)
args=parser.parse_args()
v.INPUT=args.excel.expanduser().resolve()
assert v.INPUT.is_file(),f'Raw Excel missing: {v.INPUT}'
assert args.repeats>100
coded,meta,names=v.load_codes()
N=coded.shape[0]
assert N==328
flags={name:(coded[:,j]==cat_code).astype(np.int8) for j,cat_code,name,_ in meta}
SOURCES=[
 ('07_A_Tablo4_psikososyal.csv',5),
 ('08_A_Tablo5_akademik_sosyal.csv',6),
 ('09_A_Tablo6_aile_yasam.csv',7),
]
selected=[]
for filename,table in SOURCES:
    with (args.selected_dir/filename).open(encoding='utf-8-sig',newline='') as f:
        entries=list(csv.DictReader(f))
    assert len(entries)==10,(filename,len(entries))
    selected.extend((table,i,row) for i,row in enumerate(entries,1))
assert len(selected)==30
rng=np.random.default_rng(20261120)
indices=rng.integers(0,N,size=(args.repeats,N),dtype=np.int16)
results=[]
for table,row_number,row in selected:
    antecedents=row['Oncul'].split(' + ')
    consequent=row['Sonuc']
    assert len(antecedents)==2,(table,row_number,antecedents)
    a,b,c=(flags[x] for x in (*antecedents,consequent))
    ab=a*b;abc=ab*c
    na=int(a.sum());nb=int(b.sum());nab=int(ab.sum());nc=int(c.sum());nabc=int(abc.sum())
    assert [nab,nc,nabc]==[int(row[k]) for k in ('nA','nC','nAC')]
    confidence=nabc/nab
    shorter=max(int((a*c).sum())/na,int((b*c).sum())/nb)
    delta=confidence-shorter
    assert abs(delta-float(row['Uretkenlik_delta']))<1e-12,(table,row_number)
    AA=a[indices].sum(axis=1);BB=b[indices].sum(axis=1)
    AAB=ab[indices].sum(axis=1)
    AC=(a*c)[indices].sum(axis=1);BC=(b*c)[indices].sum(axis=1)
    ABC=abc[indices].sum(axis=1)
    ok=(AA>0)&(BB>0)&(AAB>0)
    values=ABC[ok]/AAB[ok]-np.maximum(AC[ok]/AA[ok],BC[ok]/BB[ok])
    low,high=np.percentile(values,[2.5,97.5])
    results.append({'table':table,'row':row_number,'rule':row['Oncul']+' → '+row['Sonuc'],
       'nA':nab,'nC':nc,'nAC':nabc,'delta':delta,'delta_print':round(delta,3),
       'CI_low':low,'CI_high':high,'CI_low_print':round(low,3),
       'CI_high_print':round(high,3),'boot_valid':int(ok.sum())})
out=args.output.expanduser().resolve();out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
print(f'Validated N={N}; rules={len(results)}; resamples={args.repeats}; output={out}')
