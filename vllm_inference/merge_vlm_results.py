import argparse
import os
from utils.cache_util import BufferedJsonReader
import jsonlines
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str)
parser.add_argument('--stop',type=int, default=None)

args = parser.parse_args()

if args.stop is not None:
    print(f'debug mode...')
    args.output_dir='/tmp/test_syn'

# save all the results
data = []
for path in sorted(glob.glob(f"{args.output_dir}tmp_*.bin")):
    with BufferedJsonReader(path) as f:
        for x in f.read():
            data.extend(x)
data.sort(key = lambda entry: entry['syn_id']) # sort the data by entry_id
print(f"num of saved preds: {len(data)}")

out_path = f'{args.output_dir}.jsonl'
print(f'saving synthesized corpora to {out_path}')
if os.path.exists(out_path):
    os.remove(out_path)
    print(f"cached {out_path} removed successfully")
with jsonlines.open(out_path,mode='a') as writer:
    for doc in data:
        writer.write(doc)
print(f'saved jsonl to: {out_path}')

for path in glob.glob(f"{args.output_dir}tmp_*.bin"):
    os.remove(path)

print('done')
