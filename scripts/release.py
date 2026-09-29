"""Build and package one canonical manuscript; verify all delivered mirrors.

The inventory excludes itself, ZIP containers and upload_manifest.json to avoid
circular hashes. The latter records the inventory, archive and PDF hashes.
Historical audit logs are retained as historical evidence, not rerun claims.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration():
    for number,name in [(1,'main.tex'),(2,'p2_quoting.tex'),(3,'p3_forecast.tex')]:
        if (ROOT/name).exists():return number,name
    raise RuntimeError('No canonical manuscript source')


def release_files():
    extensions={'.tex','.bib','.bbl','.py','.md','.cff','.json','.png','.npz','.lean','.txt','.yml','.c'}
    out=[]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        if p.name.startswith('SUBMISSION_REVIEW_'):continue
        rel=p.relative_to(ROOT)
        if any(part in {'.git','__pycache__','.lake'} for part in rel.parts):continue
        if rel.as_posix() in {'MANIFEST_SHA256.txt','upload_manifest.json'}:continue
        if p.name in {'.gitignore','lean-toolchain'} or p.suffix in extensions or rel.as_posix()=='paper.pdf' or (rel.parts[0]=='audit' and p.suffix=='.log'):
            out.append(p)
    return sorted(out,key=lambda p:p.relative_to(ROOT).as_posix())


def write_zip(path,files,relative_root=ROOT):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:
            info=zipfile.ZipInfo(p.relative_to(relative_root).as_posix(),(2026,9,28,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16
            z.writestr(info,p.read_bytes())


def verify():
    entries=[]
    for line in (ROOT/'MANIFEST_SHA256.txt').read_text().splitlines():
        expected,name=line.split('  ',1);p=ROOT/name
        assert p.resolve().is_relative_to(ROOT) and p.is_file(),name
        assert digest(p)==expected,'Hash mismatch: '+name
        entries.append(p)
    assert {p.relative_to(ROOT).as_posix() for p in entries}=={p.relative_to(ROOT).as_posix() for p in release_files()},'Inventory coverage mismatch'
    if not (ROOT/'upload_manifest.json').exists():
        _,source=configuration()
        assert (ROOT/'paper.tex').read_text().strip().endswith('\\input{'+source+'}')
        print(f'Verified {len(entries)} delivered source files and canonical entry point.')
        return
    meta=json.loads((ROOT/'upload_manifest.json').read_text())
    archive=ROOT/meta['source_archive']
    assert digest(archive)==meta['source_archive_sha256']
    assert digest(ROOT/'paper.pdf')==meta['pdf_sha256']
    assert digest(ROOT/'MANIFEST_SHA256.txt')==meta['manifest_sha256']
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())=={p.relative_to(ROOT).as_posix() for p in entries}|{'MANIFEST_SHA256.txt'}
        for name in z.namelist():assert z.read(name)==(ROOT/name).read_bytes(),name
    if (ROOT/'verification-suite.zip').exists():
        with zipfile.ZipFile(ROOT/'verification-suite.zip') as z:
            wanted=[p for p in entries if p.parent==ROOT/'verification']
            assert set(z.namelist())=={p.name for p in wanted}
            for p in wanted:assert z.read(p.name)==p.read_bytes(),p.name
    _,main=configuration()
    assert (ROOT/'paper.tex').read_text().strip().endswith('\\input{'+main+'}')
    print(f'Verified {len(entries)} files, canonical entry point, PDF hash and every archive member.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build',action='store_true');parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:verify();return
    number,source=configuration();stem=Path(source).stem
    if args.build:
        subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',source],cwd=ROOT,check=True)
        log=(ROOT/(stem+'.log')).read_text(errors='replace')
        forbidden=['undefined references','undefined citations','multiply defined','Overfull \\hbox','Overfull \\vbox']
        assert not any(s in log for s in forbidden),'Unresolved references or overfull boxes; inspect build log'
        shutil.copyfile(ROOT/(stem+'.pdf'),ROOT/'paper.pdf')
        if (ROOT/(stem+'.bbl')).exists():shutil.copyfile(ROOT/(stem+'.bbl'),ROOT/'paper.bbl')
    files=release_files()
    inventory=ROOT/'MANIFEST_SHA256.txt'
    inventory.write_text(''.join(f'{digest(p)}  {p.relative_to(ROOT).as_posix()}\n' for p in files))
    archive='source-and-reproducibility.zip' if number==1 else 'trilogy-source-supplement.zip'
    write_zip(ROOT/archive,files+[inventory])
    if number!=1:write_zip(ROOT/'verification-suite.zip',[p for p in files if p.parent==ROOT/'verification'],ROOT/'verification')
    info=subprocess.check_output(['pdfinfo',str(ROOT/'paper.pdf')],text=True)
    pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))
    old=json.loads((ROOT/'upload_manifest.json').read_text()) if (ROOT/'upload_manifest.json').exists() else {'doi':re.search(r'10\.5281/zenodo\.\d+', (ROOT/source).read_text()).group(0)}
    metadata={'paper':number,'repository':ROOT.name,'private':False,'manuscript_date':'2026-09-28',
              'main_source':source,'pages':pages,'png_figures':len(list((ROOT/'figures').glob('*.png'))),
              'pdf_sha256':digest(ROOT/'paper.pdf'),'source_archive':archive,'source_archive_sha256':digest(ROOT/archive),
              'manifest_sha256':digest(inventory),'full_source_tree_in_repository':True,
              'original_deposit_doi':old.get('original_deposit_doi',old.get('doi')),
              'doi':old.get('doi'),
              'doi_deposit_updated':False,'data':'Synthetic mathematical experiments; no market data.'}
    (ROOT/'upload_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    verify()


if __name__=='__main__':main()
