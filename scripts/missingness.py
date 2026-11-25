from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--excel',type=Path,required=True)
parser.add_argument('--notebook',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
a=parser.parse_args()
os.environ['APR_EXCEL']=str(a.excel.resolve())
notebook=json.loads(a.notebook.read_text(encoding='utf-8'))
source='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
marker='\nA_KURALLAR,A_AUDIT=build_exploratory('
assert marker in source
namespace={'__name__':'__main__'}
exec(compile(source.split(marker,1)[0],str(a.notebook),'exec'),namespace)
df=namespace['df']; records=namespace['KAYIT_ITEMS']; AD=namespace['AD']; norm=namespace['txt']; missing=namespace['eksik']
N=len(df); assert N==350
entries=[]
for j in list(AD)+list(range(32,53)):
    name=AD[j] if j in AD else f'DASS_{j-31:02d}'
    badrows=[i for i,r in enumerate(records) if j not in r]
    n_missing=sum(missing(df.iloc[i,j]) for i in badrows)
    n_invalid=len(badrows)-n_missing
    entries.append({'Degisken':name,'Excel_sutunu':j+1,'N_temel':N,
        'Gecerli_N':N-len(badrows),'Eksik_veya_yorumlanamayan_N':len(badrows),
        'Eksik_hucre_N':n_missing,'Yorumlanamayan_N':n_invalid,
        'Eksik_oran_yuzde':round(100*len(badrows)/N,2)})
assert len(entries)==49
assert sum(r['Eksik_veya_yorumlanamayan_N'] for r in entries)==23
assert sum(r['Gecerli_N']<N for r in entries)==3
out=a.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
print('Missingness: 49 variables; 23 problem cells (3 affected variables);',out)
