import json,subprocess as s
from pathlib import Path
from urllib.parse import urlsplit
p=Path('/public/home/jiaosz/ywliang/ai4s/.sscd_periodic_self_repair_20260906_v1')
raw=s.check_output(['git','config','--get','remote.origin.url'],cwd=p,text=True).strip()
u=urlsplit(raw)
print(json.dumps({'git':s.check_output(['git','--version'],text=True).strip(),'scheme':u.scheme,'host':u.netloc.rsplit('@',1)[-1],'path':u.path,'head':s.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()}))
