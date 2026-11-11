"""Prepare only the pinned, hash-verified upstream dataset; never extract paths."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/completed/fedcpdp-paper-lopo-20261110'
URL = 'https://raw.githubusercontent.com/MeeteshNevendra/FedCPDP/ab5af637a60aed4c34a5338709b3ac65b344ec37/Dataset.zip'

def prepare(payload):
    provenance=json.loads((EXPERIMENT/'provenance.json').read_text(encoding='utf-8'))
    if hashlib.sha256(payload).hexdigest()!=provenance['dataset_archive_sha256']:
        raise ValueError('Archive SHA256 differs from the completed experiment')
    manifest=json.loads((EXPERIMENT/'manifest.json').read_text(encoding='utf-8'))
    verified={}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name, record in manifest.items():
            matches=[m for m in archive.namelist() if Path(m).name==name+'.csv']
            if len(matches)!=1: raise ValueError(f'Expected exactly one archive member for {name}')
            data=archive.read(matches[0])
            if hashlib.sha256(data).hexdigest()!=record['sha256']: raise ValueError(f'CSV hash mismatch: {name}')
            verified[name]=data
    destination=EXPERIMENT/'data'; destination.mkdir(exist_ok=True)
    # Validate the whole archive before writing any dataset files.
    for name,data in verified.items(): (destination/(name+'.csv')).write_bytes(data)
    print(f'Prepared and verified {len(verified)} CSV files in {destination}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--archive',type=Path)
    group.add_argument('--download',action='store_true',help='Explicitly retrieve the pinned third-party archive')
    args=parser.parse_args()
    if args.download:
        with urllib.request.urlopen(URL,timeout=60) as response: payload=response.read()
    else: payload=args.archive.read_bytes()
    prepare(payload)
