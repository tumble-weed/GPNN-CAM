#!/opt/conda/envs/gpnnenv/bin/python
import os
import sys
from icecream import ic
myLoc = os.path.dirname(os.path.abspath(__file__))

sessions_file = os.path.join(myLoc,'tmux_sessions.sh')
alias = sys.argv[1]
session = sys.argv[2]
ic(alias,session)
if not os.path.isfile(sessions_file):
    os.system(f'touch {sessions_file}')
with open(sessions_file,'r') as f:
    sessions = f.readlines()
found = False
for l in sessions:
    if l.startswith(f"export {alias}="):
        if l.rstrip() == f'export {alias}={session}':
            ic('found exact alias')
            found = True
            break
        else:
            assert False,(f'alias conflict, exists as {l}')
            #import ipdb;ipdb.set_trace()
if not found:
    with open(sessions_file,'a') as f:
        f.write(f'export {alias}={session}\n')



        


