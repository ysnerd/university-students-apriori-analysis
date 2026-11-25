#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, hashlib, os, shutil, subprocess, sys, time, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
S=ROOT/'scripts'
EXPECTED={'DASS_ayni_altolcek':293,'DASS_altolcekler_arasi':3232,
          'Psikososyal_alan_ici':1259,'Alanlar_arasi':2246,'Psikososyal_olmayan':6}

def rows(path,sep=','):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f,delimiter=sep))

def check(path,n):
    actual=len(rows(path))
    assert actual==n, f'{path.name}: expected {n} rows, got {actual}'
    print(f'[PASS] {path.name}: {actual} rows',flush=True)

def command(args,env=None,cwd=ROOT):
    print('RUN:', ' '.join(str(x) for x in args),flush=True)
    subprocess.run([str(x) for x in args],cwd=str(cwd),env=env,check=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'data/original.xlsx')
    ap.add_argument('--output',type=Path,default=ROOT/'results')
    ap.add_argument('--quick',action='store_true',help='Skip 499 permutations and 3000 bootstrap resamples')
    args=ap.parse_args(); t0=time.monotonic()
    data=args.data.expanduser().resolve(); out=args.output.expanduser().resolve()
    if not data.is_file():ap.error(f'Input file missing: {data}')
    print('DATA SHA256:',hashlib.sha256(data.read_bytes()).hexdigest(),flush=True)
    if out==ROOT or out==data.parent:ap.error('Output must not overwrite repository/data root')
    out.mkdir(parents=True,exist_ok=True)
    primary=out/'primary'; extras=out/'supplementary'; figures=out/'figures'
    # Bu klasörler tamamen program tarafından üretilir: yeniden çalıştırmada
    # eski hızlı/tam sonuçları karıştırmamak için yalnızca bu klasörler temizlenir.
    for folder in (primary,extras,figures):
        if folder.is_symlink():raise RuntimeError(f'Symlink output folder is not allowed: {folder}')
        if folder.exists():shutil.rmtree(folder)
        folder.mkdir(parents=True)
    # Önceki sürümün ürettiği gereksiz arşiv ve çalışma kopyalarını kaldır.
    legacy_work=out/'_work'
    if legacy_work.exists() and not legacy_work.is_symlink():shutil.rmtree(legacy_work)
    legacy_zip=out/'Apriori_Analiz_Ciktilari.zip'
    if legacy_zip.is_file():legacy_zip.unlink()
    # Ana analiz betiği kendi çalışma klasöründe bir CSV kümesi ve ZIP oluşturur.
    # Bunlar sadece GEÇİCİ dizinde üretilir, kullanıcı results/ alanına kopyalanmaz.
    env=os.environ.copy();env['APR_EXCEL']=str(data)
    required_primary={
        '01_veri_kodlama_denetimi.csv',
        '02_degisken_kategori_frekanslari_350.csv',
        '03_Ek_A1_21_DASS_maddesi.csv',
        '04_DASS21_guvenirlik.csv',
        '06_A_ana_esik_tum_kurallar.csv',
        '07_A_Tablo4_psikososyal.csv',
        '08_A_Tablo5_akademik_sosyal.csv',
        '09_A_Tablo6_aile_yasam.csv',
        '11_A_duyarlilik_esikleri.csv',
        '12_A_duyarlilik_CGPA_kardes_haric.csv',
        '13_A_istatistik_aile_ve_N.csv',
        '14_B_kisitli_ana_model_kurallari.csv',
        '15_B_kisitli_duyarlilik_tum_kurallar.csv',
        '16_B_duyarlilik_tablosu.csv',
        '17_B_test_aileleri_ve_N.csv',
        '19_A_tablo_secim_olcutleri.csv',
        '20_yorumlanamayan_yanitlar.csv',
        '21_B_besli_kategori_duyarlilik_kurallari.csv',
        '22_B_CGPA_kardes_haric_kurallar.csv',
    }
    with tempfile.TemporaryDirectory(prefix='_apriori_gecici_',dir=ROOT) as tmpdir:
        command([sys.executable,S/'main_analysis.py'],env=env,cwd=tmpdir)
        produced=Path(tmpdir)/'Apriori_Analiz_Ciktilari'
        if not produced.is_dir():raise FileNotFoundError(produced)
        for fn in sorted(required_primary):
            src=produced/fn
            if not src.is_file():raise FileNotFoundError(src)
            shutil.copy2(src,primary/fn)
    print('[PASS] Ana analiz geçici dosyaları temizlendi; yalnızca',
          len(required_primary),'gerekli ana CSV saklandı.',flush=True)
    rulefile=primary/'06_A_ana_esik_tum_kurallar.csv'
    check(rulefile,7036)
    rr=rows(rulefile)
    assert all(int(r['N'])==328 for r in rr)
    actual={g:sum(r['Sinif']==g for r in rr) for g in EXPECTED}
    assert actual==EXPECTED,(actual,EXPECTED)
    print('[PASS] 328 x 49; 7,036 rules; five classes:',actual,flush=True)
    # Conditional sample-sensitivity: same N=328 and restored N=343.
    command([sys.executable,S/'Orneklem_Duyarliligi_47_Degisken_Hizli.py',
             '--excel',data,'--notebook',S/'Apriori_Cok_Boyutlu_Analiz.ipynb',
             '--output-dir',extras,'--primary-rules',rulefile])
    check(extras/'Orneklem_47_Degisken_Ayni_N328_6973_Kural.csv',6973)
    check(extras/'Orneklem_47_Degisken_Kayit_Dahil_N343_7120_Kural.csv',7120)
    # The older 342-person alternative is written by the original core pipeline.
    alt=rows(primary/'12_A_duyarlilik_CGPA_kardes_haric.csv')
    count342=sum(float(r['Destek'])>=.15-1e-12 and float(r['Guven'])>=.70-1e-12
                 and float(r['Lift'])>=1.40-1e-12 for r in alt)
    assert count342==7144,f'Alternative N=342 expected 7,144 rules, got {count342}'
    print('[PASS] 47 variables N=342: 7,144 rules',flush=True)
    # Three additional independent / appendix computations.
    command([sys.executable,S/'DASS21_Kisitlanmis_Esik_Uygunlugu_Dogrulama.py',
             '--excel',data,'--output',extras/'Tablo_9_Lift_Esik_Uygunlugu.csv'])
    command([sys.executable,S/'DASS21_Surekli_Puan_Duyarlilik_Analizi.py',
             '--excel',data,'--output',extras/'Tablo_B3_DASS21_12_Karsilastirma.csv'])
    check(extras/'Tablo_B3_DASS21_12_Karsilastirma.csv',12)
    command([sys.executable,S/'missingness.py','--excel',data,
             '--notebook',S/'Apriori_Cok_Boyutlu_Analiz.ipynb',
             '--output',extras/'Degisken_Bazinda_Eksik_Veri_N350.csv'])
    check(extras/'Degisken_Bazinda_Eksik_Veri_N350.csv',49)
    # Figure reads the freshly computed main-rule CSV, not reference outputs.
    fenv=os.environ.copy();fenv['APR_RULES_CSV']=str(rulefile)
    fenv['APR_FIGURE_DIR']=str(figures)
    command([sys.executable,S/'Sekil_1_Alanlara_Gore_Cizim_Kodu.py'],env=fenv)
    assert (figures/'Sekil_1_Alanlara_Gore_Destek_Guven.png').stat().st_size>50000
    assert (figures/'Sekil_1_Alanlara_Gore_Destek_Guven.svg').stat().st_size>50000
    if not args.quick:
        command([sys.executable,S/'verify_perm_apriori.py',data,extras])
        perm=rows(extras/'Kural_Permutasyon_499_Ham_Tekrarlar.csv',sep=';')
        assert len(perm)==499
        assert max(int(row['TOPLAM']) for row in perm)==3
        print('[PASS] 499 marginal permutations; max 3 null rules',flush=True)
        command([sys.executable,S/'derive_table3.py',extras/'Kural_Olanaklari_Yeniden_Dogrulama.csv',extras/'Tablo_3_Kural_Olanaklari.csv'])
        command([sys.executable,S/'DeltaGuven_30_Kural_Bootstrap_Dogrulama.py',
                 '--excel',data,'--selected-dir',primary,
                 '--output',extras/'Tablo_B2_30_Kural_Bootstrap.csv','--repeats','3000'])
        b=rows(extras/'Tablo_B2_30_Kural_Bootstrap.csv')
        assert len(b)==30 and all(int(x['boot_valid'])==3000 for x in b)
        print('[PASS] 30 fixed rules; 3,000 bootstrap replicates',flush=True)
    print(f'\nSUCCESS: {"QUICK" if args.quick else "FULL"} reproducibility checks passed. Results: {out}',flush=True)
    print(f'Runtime (seconds): {time.monotonic()-t0:.1f}',flush=True)

if __name__=='__main__':main()
