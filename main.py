import argparse
import os
import sys
import time

import torch
import yaml
from loguru import logger

sys.path.append('.')
 
from BDIH import BDIH

from data.cifar100_general import CIFAR100

from utils.Dict2Obj import Dict2Obj
from utils.evaluate import test, generate_code, mean_average_precision
from utils.icarl import icarl
from torch.utils.data import DataLoader


METHODS = {'bdih': BDIH}
DATASETS = {'cifar-100_g': CIFAR100,}
def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='Hashing Template')

    parser.add_argument('--config', default='config/CIFAR100_G.yaml', type=str)

    parser.add_argument('--arch', default=None, type=str,
                        help='Backbone Type.')
    parser.add_argument('--method', default=None, type=str,
                        help='Deep Continual Hashing Method.')
    parser.add_argument('--code_length', default=None, type=int,
                        help='Length of hash code.')
    parser.add_argument('--valid_length_list', default=None, type=str,
                        help='# of valid bits in each session.')
    parser.add_argument('--step_size', default=None, type=int,
                        help='# of defrosted frezon bits.')

    parser.add_argument('--gpu', default=0, type=int,
                        help='Using gpu.')
    parser.add_argument('--project', default='Test', type=str)
    parser.add_argument('--save_checkpoint', action='store_true')

    parser.add_argument('--lr', default=None, type=float, help='')
    parser.add_argument('--lambda_kd', default=None, type=float, help='')
    parser.add_argument('--lambda_q', default=None, type=float, help='')
    
    parser.add_argument('--nf', default=1, type=float,
                        help='if use noise frozen bits.')
    parser.add_argument('--nf_ratio', default=None, type=float, 
                        help='the ratio of frozen bits being added noise.')

    parser.add_argument('--memory', default=None, type=int, 
                        help='The size of memory buffer.')

    parser.add_argument('--comments', default=None, type=str)


    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    config = Dict2Obj(config)
        
    if args.arch is not None:
        config.arch = args.arch
    if args.method is not None:
        config.method = args.method
    if args.code_length is not None:
        config.code_length = args.code_length
    if args.step_size is not None:
        config.step_size = args.step_size
    if args.valid_length_list is not None:
        config.valid_length_list = list(map(int, args.valid_length_list.split(',')))
    if config.valid_length_list is None:
        config.valid_length_list = [config.code_length - config.step_size * (config.total_session - s)  for s in range(1, config.total_session+1)]
    config.gpu = args.gpu
    if args.project is not None:
        config.project = args.project
    if args.comments is not None:
        config.comments = args.comments
    if args.save_checkpoint:
        config.save_checkpoint = True

    if args.lr is not None:
        config.lr = args.lr    
    if args.lambda_kd is not None:
        config.lambda_kd = args.lambda_kd 
    if args.lambda_q is not None:
        config.method_parameters[config.method].lambda_q = args.lambda_q  
 
    if args.nf is not None:
        config.nf = args.nf  
    if args.nf_ratio is not None:
        config.method_parameters[config.method].nf_ratio = args.nf_ratio   

    if args.memory is not None:
        config.memory = args.memory 

    return config

if __name__ == '__main__':
    torch.set_num_threads(1)
    config = load_config()
    timestr = time.strftime('%Y-%m-%d-%H:%M', time.gmtime())
    save_name = '{}bits_{}'.format(config.code_length, timestr)
    
    logger.add(os.path.join('Results', config.project, 'logs', config.method, config.dataset, save_name+'.log'), rotation='500 MB', level='INFO')
    
    #! Output configs
    logger.info('--------------------------Current Settings--------------------------')
    config.method_parameters = config.method_parameters[config.method]
    for key, value in config.items():
        logger.info('{} = {}'.format(key, value))
    
    #! Set device
    config.device = torch.device('cpu') if config.gpu is None else torch.device('cuda:{}'.format(config.gpu))
    
    #! Set dataset
    dataset = DATASETS[config.dataset]

    TOTAL_SESSION = config.total_session
    num_class_list = config.num_class_list

    #! Traing
    for s in range(TOTAL_SESSION):
        session_id = s
        logger.info('------------Session {} Training Start-------------------------------'.format(session_id))
        
        # Load data
        ### load gallery data
        eval_trainset = dataset(mode="gallery", session_id=session_id)
        eval_trainloader = DataLoader(eval_trainset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        ### load testing query data
        testset = dataset(mode="test",session_id=session_id)
        testloader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        ### load validation query data
        valset = dataset(mode="val",session_id=session_id)
        valloader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        ### load training data
        trainset = dataset(mode="train",session_id=session_id)
        if session_id != 0 and config.replay:
            trainset.add_memory(prev_exem_names, prev_exem_labels)
            
        trainloader = DataLoader(trainset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, drop_last=True)

        # Load method
        if session_id == 0:
            method = METHODS[config.method](config)
        else:
            prev_model = best_model
            method.update(session_id=session_id, 
                            old_model=prev_model, 
            )
        best_mAP = 0.0
        max_iters = method.max_iters
        for i in range(max_iters):    
            if session_id == 0:
                model = method.train_iter(trainloader, i)
            else:
                model = method.inc_train_iter(trainloader, i)

            # Test with val
            if i % config.val_every_n_epoch == 0 or i == max_iters-1:
                mAP, _, _ = test(
                    model                = model,
                    query_dataloader     = valloader,
                    retrieval_dataloader = eval_trainloader,
                    code_length          = config.valid_length_list[session_id],
                    topk                 = config.topk,
                    device               = config.device,
                )
                if mAP > best_mAP:
                    best_mAP = mAP
                    best_model = model
                logger.info('[Session:{}][it:{}/{}][Current MAP:{:.4f}][Best Session MAP:{:.4f}]'.format(session_id, i+1, max_iters, mAP, best_mAP))
        
        logger.info('------------Session {} Training End---------------------------------'.format(session_id))
        # Test with test
        if session_id == 0:
            cur_valid_length = config.valid_length_list[0]
            query_codes = generate_code(best_model, testloader, cur_valid_length, config.device)
            prev_gallery_codes = generate_code(best_model, eval_trainloader, cur_valid_length, config.device)
            query_targets = testloader.dataset.get_onehot_targets().to(config.device)
            retrieval_targets = eval_trainloader.dataset.get_onehot_targets().to(config.device)
            mAP = mean_average_precision(
                query_codes.to(config.device),
                prev_gallery_codes.to(config.device),
                query_targets.to(config.device),
                retrieval_targets.to(config.device),
                config.device,
                config.topk,
            )
            prev_gallery_targets = retrieval_targets
            prev_testloader = testloader
            prev_mAP = 0.0

        else:
            cur_valid_length = config.valid_length_list[session_id]
            prev_query_codes = generate_code(best_model, prev_testloader, cur_valid_length, config.device)
            prev_query_targets = prev_testloader.dataset.get_onehot_targets().to(config.device)

            query_codes = generate_code(best_model, testloader, cur_valid_length, config.device)
            query_targets = testloader.dataset.get_onehot_targets().to(config.device)

            inc_gallery_codes = generate_code(best_model, eval_trainloader, cur_valid_length, config.device)
            inc_gallery_targets = eval_trainloader.dataset.get_onehot_targets().to(config.device)

            # expand prev_gallery_codes
            append_code = torch.ones([prev_gallery_codes.shape[0], cur_valid_length-prev_gallery_codes.shape[1]]) * 1
            prev_gallery_codes = torch.cat((prev_gallery_codes.cpu(), append_code), 1)
            overall_retrieval_code = torch.vstack((inc_gallery_codes.to(config.device), prev_gallery_codes.to(config.device)))
            overall_retrieval_targets = torch.vstack((inc_gallery_targets.to(config.device), prev_gallery_targets.to(config.device)))

            prev_mAP = mean_average_precision(
                query_code=prev_query_codes.to(config.device),
                retrieval_code=overall_retrieval_code.to(config.device),
                query_targets=prev_query_targets,
                retrieval_targets=overall_retrieval_targets,
                device=config.device,
                topk=config.topk
            )
            mAP = mean_average_precision(
                query_code=query_codes.to(config.device),
                retrieval_code=overall_retrieval_code.to(config.device),
                query_targets=query_targets,
                retrieval_targets=overall_retrieval_targets,
                device=config.device,
                topk=config.topk
            )
            prev_gallery_codes = overall_retrieval_code
            prev_gallery_targets = overall_retrieval_targets
        logger.info('[Session:{}][Total Test MAP:{:.4f}][Prev Test MAP:{:.4f}]'.format(session_id, mAP, prev_mAP))
        # add replay data to memory
        if config.replay:
            best_model.eval()
            if session_id>0:
                eval_trainset.add_memory(prev_exem_names,prev_exem_labels) 
            prev_exem_names, prev_exem_labels = [], []
            exemplar_set = icarl(best_model, eval_trainset, config.memory, config.device) # 返回一个字典,key为class id, value为image
            for cat in exemplar_set:
                prev_exem_names.extend(exemplar_set[cat])
                prev_exem_labels.extend([cat]*len(exemplar_set[cat]))
        if config.save_checkpoint:
            save_dir = os.path.join('Results', config.project, 'checkpoints', config.method, config.dataset, save_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir) 
            torch.save(best_model.cpu(), os.path.join(save_dir, 'model_session{}.t'.format(s)))
    if config.save_checkpoint:
        torch.save(method.hashcenters.cpu(), os.path.join(save_dir, 'centers.t'))
