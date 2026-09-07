import argparse,base64,hashlib,json,zlib
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('receipt',type=Path)
parser.add_argument('--output-dir', type=Path)
args=parser.parse_args()
receipt=json.loads(args.receipt.read_text(encoding='utf-8'))
assert receipt['status']=='completed' and receipt['returncode']==0
payload=json.loads(receipt['output'])
target=args.output_dir or Path(__file__).resolve().parents[2]/'docs/r03_paper_story_20260907/execution/assets/canary_40403'
target.mkdir(exist_ok=True)
records=[]
for name,row in payload.items():
 assert Path(name).name==name and name.endswith(('.json','.jsonl'))
 data=zlib.decompress(base64.b64decode(row['zlib_base64']))
 assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
 path=target/name
 if path.exists():assert path.read_bytes()==data
 else:path.write_bytes(data)
 records.append({k:v for k,v in row.items() if k!='zlib_base64'}|{'local':str(path)})
(target/'SOURCE_RECEIPTS.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
print(json.dumps(records))
