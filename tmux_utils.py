import subprocess
import os
import time
def kill_all_tmux(group_name):
    with open(f'{group_name}-sessions','r') as f:
        for line in f:
            session_name = line.strip()
            subprocess.run(f"tmux kill-session -t {session_name}",shell=True)

def view_all_logs(group_name):
    log_names = []
    with open(f'{group_name}-tmux-logs','r') as f:
        for line in f:
            log_name = line.strip()
            log_names.append(log_name)
            # subprocess.run(f"tmux attach -t {log_name}",shell=True)
    while True:
        command=''
        for log_name in log_names:
            command += f'echo "{log_name}\n"; tail -n 7 {log_name}; printf "\n\n";'
            # os.system(f'tail -n 5 {log_name}')
            # print('\n'*2)
        # clear screen and rerun command in endless loop
        os.system(command)
        time.sleep(2)
        os.system('clear')
def get_method_model_alias(session_name):
    methods_and_alias = {
                'scorecam':'s',
               'relevancecam':'r',
               'gradcam':'g',
               'gradcampp':'gpp',
               'gpnn-gradcam':'gg',
               'gpnn-gradcampp':'ggpp',
               'gpnn-relevancecam':'gr',
               'gpnn-gradcam-uniform':'ggu',
               'cameras':'c',
               'layercam':'l',
    }
    models_and_alias = {
            'vgg16':'16',
              'resnet50':'50',
    }
    # detect which method in session_name:
    #-------------------------------------------------------------
    import itertools
    methods = methods_and_alias.keys()
    method_alias = methods_and_alias.values()
    models = models_and_alias.keys()
    model_alias = models_and_alias.values()
    method_model_combinations = itertools.product(methods,models)
    alias_combinations = itertools.product(method_alias,model_alias)
    found = None
    alias_found = None
    for mm,alias in zip(method_model_combinations,alias_combinations):
        # method,model = mm
        # if method in session_name and model in session_name:
        #     return method,model
        mm = '-'.join(mm)
        alias = ''.join(alias)
        # print(mm,session_name)
        if mm in session_name:
            # assert found == None
            if found !=None:
                if len(mm) < len(found):
                    continue
            found = mm
            alias_found = alias
    # import ipdb;ipdb.set_trace()
    return alias_found
    #-------------------------------------------------------------

def create_alias(session_name):
    # print(f"alias {session_name}='tmux attach -t {session_name}'")
    alias = ''
    if 'pascal' in session_name:
        alias += 'p'
    if session_name.startswith('t-metrics'):
        # session_name = session_name[2:]
        # def get_metrics_alias(session_name):
        alias = 'm'
        if 'chattopadhyay' in session_name:
            alias += 'c'
        elif 'perturbation' in session_name:
            alias += 'p'
        mm_alias = get_method_model_alias(session_name)    
        alias += mm_alias
        
    else:
        mm_alias = get_method_model_alias(session_name)
        alias += mm_alias
    print(alias)
    return alias
def create_aliases(group_name):
    sessions_filename = f'{group_name}-sessions'
    alias_commands= []
    with open (sessions_filename,'r') as f:
        for line in f:
            session_name = line.strip()
            alias = create_alias(session_name)
            alias_command = f"export {alias}={session_name}\n"
            alias_commands.append(alias_command)
            # print(f"alias {session_name}='tmux attach -t {session_name}'")
    # assert False
    with open(f"{group_name}-aliases",'w') as f:
        for c in alias_commands:
            f.write(c)
            

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--kill', action='store_true',default=False)
parser.add_argument('--view', action='store_true',default=False)
parser.add_argument('--alias', action='store_true',default=False)
parser.add_argument('--group', type=str, default=None)
args = parser.parse_args()
if args.kill:
    kill_all_tmux(args.group)
if args.view:
    view_all_logs(args.group)
if args.alias:
    create_aliases(args.group)