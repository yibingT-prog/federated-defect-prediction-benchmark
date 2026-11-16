"""Rebuild frozen ECFML/FedDP data from a pinned archive; do not redistribute CSVs."""
import argparse,hashlib,io,json,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URL='https://raw.githubusercontent.com/MeeteshNevendra/FedCPDP/ab5af637a60aed4c34a5338709b3ac65b344ec37/Dataset.zip'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',required=True,choices=['ecfml','feddp'])
    g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--archive',type=Path);g.add_argument('--download',action='store_true');args=ap.parse_args()
    exp=ROOT/'experiments/homogeneous'/f'{args.method}-resource-20261116'
    if args.download:
        with urllib.request.urlopen(URL,timeout=60) as response:payload=response.read()
    else:payload=args.archive.read_bytes()
    manifest=json.loads((exp/'data_manifest.json').read_text(encoding='utf-8'));verified={}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name,record in manifest.items():
            members=[n for n in archive.namelist() if Path(n).name==name+'.csv'];assert len(members)==1,name
            data=archive.read(members[0]);assert hashlib.sha256(data).hexdigest()==record['sha256'],name;verified[name]=data
    (exp/'data').mkdir(exist_ok=True)
    for name,data in verified.items():(exp/'data'/(name+'.csv')).write_bytes(data)
    print('PASS:',args.method,len(verified),'hash-verified project files')

if __name__=='__main__':main()
