# methods=("gradcam" "gradcampp" "relevancecam")
# models=("vgg16" "resnet50")
# METRIC="chattopadhyay"
# SKIP="false"
import os
from collections import defaultdict
def run_many_metrics(group_name, datasets,methods, models, metricnames, device_flags,skip_flags, ignore_flags,median_k=41):
    session_names = []
    for dataset in datasets:
        for method in methods:
            for model in models:
                for metricname in metricnames:
                    
                    dataset_stub = f"{dataset}-" if dataset else ""
                    worm = f'{dataset}{metricname}-{method}-{model}'
                    session_name = f't-metrics-{worm}'
                    if worm in ignore_flags:
                        continue
                    device = device_flags[worm]
                    skip = skip_flags[worm]
                    # metricpy = "run_metrics_librecam"
                    metricpy = "run_metrics_librecam2"
                    # if dataset == "voc":
                    #     metricpy = "pascal_run_metrics_librecam"
                    
                    # command = f"CUDA_VISIBLE_DEVICES={device} python -m benchmark.{metricpy} --method {method} --model {model} --metric {metricname} --dataset {dataset_stub} --skip {skip}"
                    
                    command = f"source ~/.bashrc;CUDA_VISIBLE_DEVICES={device} python -m benchmark.{metricpy} --method {method} --model {model} --metric {metricname} --skip {skip} --dataset {dataset} --median_k {median_k}"
                    # import ipdb; ipdb.set_trace()
                    os.system(f'rm {session_name}.tmux.log')
                    command = f"{command} | tee -a {session_name}.tmux.log"
                    command = f'{command};bash'
                    # command = f"{command}| trap ERR bash"
                    print(command)
                    if True:
                        # import ipdb; ipdb.set_trace()
                        if os.environ.get('RUN_MANY_DRY_RUN',False) == '1':
                            print("SKIPPING ACTUALLY CREATING SESSION")
                            print(f'{session_name}, {command}')
                        else:
                            os.system(f"tmux kill-session -t {session_name}")
                            os.system(f"tmux new-session -d -s {session_name} '{command}'")                
                    
                    session_names.append(session_name)
                    
    with open(f'{group_name}-tmux-logs','w') as f:
        for session_name in session_names:
            f.write(f'{session_name}.tmux.log\n')
    with open(f'{group_name}-sessions','w') as f:
        for session_name in session_names:
            f.write(f'{session_name}\n')
    return session_names                

#======================================================================
def main():
    group_name = 'metrics-c'
    methods = [ 
            #    "gradcam" , 
            #    "gradcampp" , 
            #    "relevancecam", 
            #    "scorecam" , 
            "gpnn-gradcam", "gpnn-gradcampp", 
            #    "gpnn-gradcam-uniform", "gpnn-gradcampp-uniform"
            ]
    models = [ 
            "vgg16" , 
            #   "resnet50" 
            ]
    metricnames = [ 
                "chattopadhyay", 
                #    "perturbation"
                ]
    datasets = [ 
                # "pascal" , 
                ""
                ]
    # skip_flags = { 
    #               "gradcam-vgg16" : "false" , "gradcam-resnet50" : "false" ,
    #               "gradcampp-vgg16" : "false" , "gradcampp-resnet50" : "false" ,
    #               "relevancecam-vgg16" : "false" , "relevancecam-resnet50" : "false" ,
    #               "scorecam-vgg16" : "false" , "scorecam-resnet50" : "false" ,
    #               "gpnn-gradcam-vgg16" : "false" , "gpnn-gradcam-resnet50" : "false" ,
    #               "gpnn-gradcampp-vgg16" : "false" , "gpnn-gradcampp-resnet50" : "false" ,
    #               "gpnn-gradcam-uniform-vgg16" : "false" , "gpnn-gradcam-uniform-resnet50" : "false" ,
    #               "gpnn-gradcampp-uniform-vgg16" : "false" , "gpnn-gradcampp-uniform-resnet50" : "false" ,}
    skip_flags = defaultdict(lambda : "false")
    # device_flags = {
    #     "gradcam-vgg16" : "0" , "gradcam-resnet50" : "0" ,
    #     "gradcampp-vgg16" : "0" , "gradcampp-resnet50" : "0" ,
    #     "relevancecam-vgg16" : "0" , "relevancecam-resnet50" : "0" ,
    #     "scorecam-vgg16" : "0" , "scorecam-resnet50" : "0" ,
    #     "gpnn-gradcam-vgg16" : "0" , "gpnn-gradcam-resnet50" : "0" ,
    #     "gpnn-gradcampp-vgg16" : "0" , "gpnn-gradcampp-resnet50" : "0" ,
    #     "gpnn-gradcam-uniform-vgg16" : "0" , "gpnn-gradcam-uniform-resnet50" : "0" ,
    #     "gpnn-gradcampp-uniform-vgg16" : "0" , "gpnn-gradcampp-uniform-resnet50" : "0" ,}
    device_flags = defaultdict(lambda : "0")
    # ignore_flags = [ 
    #                 "gpnn-gradcampp-uniform-vgg16" , "gpnn-gradcampp-uniform-resnet50" 
    #                 ]
    class ignore_flags_:
        def __contains__(self, item):
            return "gpnn-gradcampp-uniform" in item
    ignore_flags=ignore_flags_()
    session_names = run_many_metrics(group_name, datasets,methods, models, metricnames, device_flags,skip_flags, ignore_flags)


    # def create_alias(session_name):
    #     alias=""
    #     if 'scorecam' in session_name:
    #         alias=f"s{alias}"
    #     elif 'scorecam' in session_name:
    #         alias=f"s{alias}"        
    #     elif 'scorecam' in session_name:
    #         alias=f"s{alias}"        
    #     elif 'scorecam' in session_name:
    #         alias=f"s{alias}"        
                            
    #     if 'vgg16' in session_name:
    #         alias=f"{alias}16"
    #     elif 'resnet50' in session_name:
    #         alias=f"{alias}50"
    #     alias_command = f'{alias}={session_name}'
    #     # print(alias_command)
    #     return alias_command


    # print(session_names)
    # with open(f'{group_name}-aliases','w') as f:
    #     for session_name in session_names:
    #         f.write(f'{create_alias(session_name)}\n')

    from tmux_utils import create_aliases
    create_aliases(group_name)
if __name__ == '__main__':
    main()