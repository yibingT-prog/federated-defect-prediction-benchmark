"""Reconstruct all six frozen data copies only after validating every CSV."""
import argparse,hashlib,io,json,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/privacy/fedcpdp-native-20261120'
URL='https://raw.githubusercontent.com/MeeteshNevendra/FedCPDP/ab5af637a60aed4c34a5338709b3ac65b344ec37/Dataset.zip'
def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--archive',type=Path);g.add_argument('--download',action='store_true');a=ap.parse_args()
    if a.download:
        with urllib.request.urlopen(URL,timeout=60) as r:payload=r.read()
    else:payload=a.archive.read_bytes()
    conditions=json.loads((EXP/'study.json').read_text())['conditions'];verified={}
    manifest=json.loads((EXP/'conditions'/conditions[0]/'data_manifest.json').read_text())
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for name,meta in manifest.items():
            matches=[n for n in z.namelist() if Path(n).name==name+'.csv'];assert len(matches)==1,name
            data=z.read(matches[0]);assert hashlib.sha256(data).hexdigest()==meta['sha256'],name
            verified[name]=data
    for c in conditions:
        assert json.loads((EXP/'conditions'/c/'data_manifest.json').read_text())==manifest
    for c in conditions:
        dest=EXP/'conditions'/c/'data';dest.mkdir(exist_ok=True)
        for name,data in verified.items():(dest/(name+'.csv')).write_bytes(data)
    print('PASS: 30 validated project files prepared for six privacy conditions')
if __name__=='__main__':main()
