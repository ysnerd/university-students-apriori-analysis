from pathlib import Path
import os
from zipfile import ZipFile
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parent
CSV_NAME = '06_A_ana_esik_tum_kurallar.csv'
ZIP = ROOT / 'ABC_APRIORI_HAKEM_CIKTILARI.zip'
OUT_DIR = Path(os.environ['APR_FIGURE_DIR']) if os.environ.get('APR_FIGURE_DIR') else ((ROOT.parent / 'Ek Analizler') if ROOT.name == 'Kodlar' else ROOT)
OUT_DIR.mkdir(parents=True, exist_ok=True)
PNG = OUT_DIR / 'Sekil_1_Alanlara_Gore_Destek_Guven.png'
SVG = OUT_DIR / 'Sekil_1_Alanlara_Gore_Destek_Guven.svg'

locations = ([Path(os.environ['APR_RULES_CSV'])] if os.environ.get('APR_RULES_CSV') else []) + [ROOT / CSV_NAME, ROOT.parent / 'Çıktılar' / CSV_NAME,
             ROOT / 'Çıktılar' / CSV_NAME]
source_csv = next((path for path in locations if path.is_file()), None)
if source_csv is not None:
    df = pd.read_csv(source_csv)
elif ZIP.is_file():
    with ZipFile(ZIP) as z:
        df = pd.read_csv(io.BytesIO(z.read(CSV_NAME)))
else:
    raise FileNotFoundError('Tam kural CSV dosyası Çıktılar/ altında veya kaynak ZIP içinde bulunamadı')

class_groups = [
    ('DASS_ayni_altolcek', 'Aynı DASS-21 alt ölçeği'),
    ('DASS_altolcekler_arasi', 'Farklı DASS-21 alt ölçekleri'),
    ('Psikososyal_alan_ici', 'Diğer psikososyal örüntüler'),
    ('Alanlar_arasi', 'Psikososyal ve diğer alanlar'),
    ('Psikososyal_olmayan', 'Psikososyal olmayan alanlar'),
]
expected = [293, 3232, 1259, 2246, 6]
actual = [int((df['Sinif'] == c).sum()) for c, _ in class_groups]
assert actual == expected, (actual, expected)
assert len(df) == 7036
assert set(df['Sinif']) == {v[0] for v in class_groups}
assert df.N.eq(328).all()

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 13.5,
    'axes.titleweight': 'semibold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'svg.fonttype': 'none',
    'savefig.facecolor': 'white',
})
fig, grid = plt.subplots(3, 2, figsize=(11.7, 10.6), dpi=280, sharex=False, sharey=False)
fig.subplots_adjust(left=.105, right=.84, top=.865, bottom=.13, hspace=.49, wspace=.17)
fig.suptitle('Keşifsel birliktelik kuralları: değişken alanlarına göre dağılım',
             x=.49, y=.974, fontsize=17, fontweight='bold')
fig.text(.49, .931, 'N = 328  |  Toplam 7.036 kural  |  Ortak destek ve güven ölçekleri',
         ha='center', va='center', color='#566273', fontsize=11.5)

cmap = plt.colormaps['viridis']
norm = Normalize(vmin=1.4, vmax=2.42)
letters = 'ABCDE'
for ax, ((value, label), n, letter) in zip(grid.flat[:5], zip(class_groups, actual, letters)):
    part = df[df['Sinif'] == value]
    ax.set_facecolor('#fbfcfd')
    ax.set_xlim(.14, .46)
    ax.set_ylim(.69, 1.01)
    ax.set_xticks([.15, .25, .35, .45])
    ax.set_yticks([.70, .80, .90, 1.00])
    ax.grid(which='major', color='#e2e7ec', linewidth=.8, zorder=0)
    ax.scatter(part['Destek'], part['Guven'], c=part['Lift'],
               cmap=cmap, norm=norm, s=14 if n != 6 else 32,
               alpha=.55 if n != 6 else .95,
               edgecolors='none' if n != 6 else '#455164',
               linewidths=0 if n != 6 else .6,
               rasterized=True, zorder=2)
    for spine in ax.spines.values():
        spine.set_color('#aab4bf')
        spine.set_linewidth(.8)
    ax.set_title(f'{letter}  {label}\n(n = {n:,})'.replace(',', '.'),
                 pad=8, loc='left', fontsize=13)

ax = grid.flat[5]
ax.set_facecolor('#fbfcfd')
short = ['Farklı DASS alt ölçekleri', 'Psikososyal ve diğer alanlar',
         'Diğer psikososyal', 'Aynı DASS alt ölçeği', 'Psikososyal olmayan']
share_count = [3232, 2246, 1259, 293, 6]
share = [100*n/len(df) for n in share_count]
y = np.arange(len(short))
ax.set_title('F  Toplam kurallar içindeki pay', loc='left', fontsize=13, pad=8)
ax.barh(y, share, color='#8696aa', alpha=.85, height=.63, zorder=2)
ax.set_yticks([])
for ii, label in enumerate(short):
    ax.text(-41.5, ii, label, fontsize=9.8, va='center', ha='left', color='#32465c')
ax.invert_yaxis()
ax.set_xlim(-43, 58)
ax.set_xticks([0, 20, 40], labels=['%0', '%20', '%40'])
ax.grid(axis='x', color='#e2e7ec', linewidth=.8, zorder=0)
ax.spines[['top','right','left']].set_visible(False)
ax.spines['bottom'].set_color('#aab4bf')
ax.tick_params(axis='y', length=0)
for i, amount in enumerate(share):
    ax.text(amount + 1, i, ('%'+f'{amount:.1f}').replace('.', ','),
            va='center', ha='left', fontsize=10.5, color='#32465c')
fig.supxlabel('Destek (support)', x=.49, y=.073, fontsize=13)
fig.supylabel('Güven (confidence)', x=.032, y=.53, fontsize=13)
cax = fig.add_axes([.875, .315, .021, .43])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label('Lift', fontsize=12, labelpad=10)
cb.set_ticks([1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
cb.ax.tick_params(labelsize=10)
fig.text(.49, .027,
         'Eşikler: destek ≥ 0,15; güven ≥ 0,70; lift ≥ 1,40. '
         'Paneller birbirini dışlayan kural gruplarını gösterir.',
         ha='center', va='center', fontsize=10.5, color='#526173')
fig.savefig(PNG, dpi=280)
fig.savefig(SVG)
print('PNG:',PNG,'SVG:',SVG, 'Group counts:',actual)
print('Ranges:',df[['Destek','Guven','Lift']].agg(['min','max']).round(4).to_dict())
