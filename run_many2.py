import os
import colorful
import threading
# PYTHON = 'python'
def run_many(group_name,datasets,base_methodnames,modelnames,device_flags,skip_flags,ignore_flags,DEBUG,start_flags = {},env_flags={},run_metric_flags={},purge_metric_flags={},
             skip_incorrect_flags={}):
    #======================================================
    str_modelnames = ' '.join(modelnames)
    session_names=[]
    for dataset in datasets:
        for base_methodname in base_methodnames:
            # for modelname in modelnames:
                    dataset_stub = dataset+"-" if dataset is "pascal" else ""
                    # worm=f"{dataset_stub}{base_methodname}-{modelname}"
                    worm = f"{dataset_stub}{base_methodname}"
                    start_worm = start_flags.get(worm,'')
                    
                    if ignore_flags.get(worm,None) is not None:
                        print("continuing")
                        continue
                    print(colorful.red('hardcoding run file to pascal_run_competing_saliency_librecam'))
                    # saliency_file = 'benchmark.run_competing_saliency_librecam '
                    # if dataset == 'pascal':
                    #     saliency_file = 'benchmark.pascal_run_competing_saliency_librecam '
                    saliency_file = 'benchmark.pascal_run_competing_saliency_librecam2 '
                    session_name=f"t-{worm}{start_worm}"
                    print(session_name)
                    # import ipdb;ipdb.set_trace()
                    skip=skip_flags[worm]
                    skip_incorrect = skip_incorrect_flags[worm]
                    device=device_flags[worm]
                    
                    start = start_flags.get(worm,None)
                    run_metric = None
                    if run_metric_flags.get(worm,None) is not None:
                        run_metric = run_metric_flags[worm]
                    purge_metric = None
                    if purge_metric_flags.get(worm,None) is not None:
                        purge_metric = purge_metric_flags[worm]
                    # import ipdb;ipdb.set_trace()
                    # command = "source ~/.bashrc;"
                    command = ""
                    command = command + " ".join([f"{k}={v}" for k,v in env_flags.items()])
                    TRACE_STEP_BY_STEP = os.environ.get('TRACE_STEP_BY_STEP','')
                    # import ipdb; ipdb.set_trace()
                    command=command + " " + f'TRACE_STEP_BY_STEP={TRACE_STEP_BY_STEP} CUDA_VISIBLE_DEVICES={device}  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024  python -m {saliency_file} --methodnames {base_methodname} --modelnames {str_modelnames} --skip {skip} --dataset {"imagenet" if dataset == "" else dataset} --skip_incorrect {skip_incorrect}'
                    if run_metric is not None:
                        command = f"{command} --run_metric {run_metric}"
                    if purge_metric is not None:
                        command = f"{command} --purge_metric {purge_metric}"
                    if start:
                        command = f"{command} --start {start}"
                    if DEBUG:
                        command=f"bash log_tmux $session_name & {command}"
                        command=f"export BREAK_COMPETING=10 && {command}"
                        print(command)
                        #  exit 1
                        # import ipdb;ipdb.set_trace()
                    else:
                        os.system(f'rm {session_name}.tmux.log')
                        # command=f"{command} | tee -a {session_name}.tmux.log"
                        # command=f"{command};trap ERR bash"
                        command=f"{command};bash"
                        print(command)
                        
                        if not os.environ.get('RUN_MANY_DRY_RUN',False) == '1':
                            # assert False
                            os.system(f"tmux kill-session -t {session_name}")
                            print(command)
                            os.system(f"tmux new-session -d -s {session_name} '{command}'")
                        else:
                            print("SKIPPING ACTUALLY CREATING SESSION")
                            print(f'{session_name}, {command}')
                        # alias=""
                        session_names.append(session_name)

    # os.system(f'rm {group_name}-tmux-logs')
    # os.system(f'rm {group_name}-sessions'
    #           )
    with open(f'{group_name}-tmux-logs','w') as f:
        for session_name in session_names:
            f.write(f'{session_name}.tmux.log\n')
    with open(f'{group_name}-sessions','w') as f:
        for session_name in session_names:
            f.write(f'{session_name}\n')
    return session_names

