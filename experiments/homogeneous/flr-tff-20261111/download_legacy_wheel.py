import base64
import hashlib
from pathlib import Path
import time
import urllib.request
import zipfile

url = 'https://storage.googleapis.com/jax-releases/nocuda/jaxlib-0.1.76-cp38-none-manylinux2010_x86_64.whl'
root = Path(__file__).resolve().parent/'wheelhouse'
root.mkdir(exist_ok=True)
dest = root/url.rsplit('/',1)[-1]
head = urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30)
size = int(head.headers['Content-Length'])
hash_header = head.headers.get('x-goog-hash','')
print('Expected',size,hash_header,flush=True)
for attempt in range(20):
    offset = dest.stat().st_size if dest.exists() else 0
    if offset == size: break
    headers = {'Range':'bytes='+str(offset)+'-'} if offset else {}
    try:
        response = urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60)
        if offset and response.status != 206:
            raise RuntimeError('Server did not honor resume range')
        if offset:
            assert response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-')
        with dest.open('ab' if offset else 'wb') as out:
            while True:
                chunk = response.read(1024*1024)
                if not chunk: break
                out.write(chunk)
        print('Downloaded',dest.stat().st_size,'/',size,flush=True)
    except Exception as exc:
        print('Retry',attempt,type(exc).__name__,str(exc),flush=True)
        time.sleep(2)
assert dest.stat().st_size == size
raw = dest.read_bytes()
md5 = base64.b64encode(hashlib.md5(raw).digest()).decode()
if 'md5=' in hash_header: assert 'md5='+md5 in hash_header
with zipfile.ZipFile(dest) as z: assert z.testzip() is None
print('VERIFIED SHA256',hashlib.sha256(raw).hexdigest(),flush=True)
