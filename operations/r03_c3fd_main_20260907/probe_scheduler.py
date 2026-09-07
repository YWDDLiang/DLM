import json,os,subprocess
out={}
for key,args in [('queue',['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%D|%b|%C|%M|%l']),('nodes',['sinfo','-h','-p','gpu','-o','%N|%t|%G|%C'])]:
 r=subprocess.run(args,capture_output=True,text=True,timeout=15)
 out[key]={'code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps(out))
