"""Fetch the pinned author archive and verify all 30 CSVs before writing."""
import argparse,hashlib,io,json,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/homogeneous/almity-svm-20261114'
URL='https://raw.githubusercontent.com/MeeteshNevendra/FedCPDP/ab5af637a60aed4c34a5338709b3ac65b344ec37/Dataset.zip'

def main():
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--archive',type=Path);group.add_argument('--download',action='store_true');args=ap.parse_args()
    if args.download:
        with urllib.request.urlopen(URL,timeout=60) as response: payload=response.read()
    else: payload=args.archive.read_bytes()
    manifest=json.loads((EXP/'data_manifest.json').read_text());verified={}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name,record in manifest.items():
            members=[n for n in archive.namelist() if Path(n).name==name+'.csv']
            assert len(members)==1,name
            data=archive.read(members[0]);assert hashlib.sha256(data).hexdigest()==record['sha256'],name
            verified[name]=data
    (EXP/'data').mkdir(exist_ok=True)
    for name,data in verified.items():(EXP/'data'/(name+'.csv')).write_bytes(data)
    print('PASS: prepared',len(verified),'hash-verified CSV files')
if __name__=='__main__':main()
