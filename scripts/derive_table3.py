import argparse,csv
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('output',type=Path)
a=ap.parse_args()
with a.source.open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f,delimiter=';'))
assert len(rr)==5
nums=[]
for x in rr:
    n=int(x['Uygun aday']);s=int(x['Destek_015_saglayan']);r=int(x['Tüm_esikler'])
    nums.append([x['Kural grubu'],n,s,r])
nums.append(['TOPLAM',sum(x[1] for x in nums),sum(x[2] for x in nums),sum(x[3] for x in nums)])
assert nums[-1][1:]==[360625,34265,7036]
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f,delimiter=';');w.writerow(['Kural grubu','Uygun aday hipotez (1 veya 2 öncül)','Destek >= 0.15 sağlayan aday','Tüm eşikleri geçen kural','Oran adayda (%)','Oran destekli adayda (%)'])
    for g,n,s,r in nums:w.writerow([g,n,s,r,round(100*r/n,3),round(100*r/s,3)])
print('Created candidate-rule comparison table:',a.output)
