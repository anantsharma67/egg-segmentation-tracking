import sys, time, json
import classical as c
name=sys.argv[1]
fix=lambda l:[p for p in l]
dev=fix(c.SPLITS['dev'])
if len(sys.argv)>2: dev=dev[:int(sys.argv[2])]
t=time.time()
r=c.evaluate(getattr(c,name),dev)
print(f'{name} n={r[4]} mIoU={r[0]:.3f} medIoU={r[1]:.3f} Dice={r[2]:.3f} count-acc={r[3]:.2f} ({time.time()-t:.0f}s)')
