import os
import pickle
import argparse
import logging

import torch
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB

from experiments.regression.hyperparams.worker import create_SpiralsWorker, create_FlightWorker, create_UCIWorker
from src.utils import mkdir

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='HpBandSter script')
parser.add_argument('--min_budget', type=float, help='minimum number of epochs for training (default: 100)',
                    default=100)
parser.add_argument('--max_budget', type=float, help='maximum number of epochs for training (default: 1000)',
                    default=1000)
parser.add_argument('--n_iterations', type=int, help='number of iterations performed by the optimizer (default: 10)',
                    default=10)
parser.add_argument('--run_id', type=str, help='a unique run id for this optimization run')
parser.add_argument('--nic_name', type=str, help='which network interface to use for communication (default: lo)',
                    default='lo')
parser.add_argument('--dataset', help='toggles which dataset to optimize for (default: spirals)',
                    choices=['spirals', 'boston', 'concrete',
                             'energy', 'power', 'wine', 'yacht',
                             'kin8nm', 'naval', 'protein', 'boston_gap',
                             'concrete_gap', 'energy_gap', 'power_gap',
                             'wine_gap', 'yacht_gap', 'kin8nm_gap',
                             'naval_gap', 'protein_gap', 'flights'], default='spirals')
parser.add_argument('--n_split', type=int, help='id of split to use (default: 0)', default=0)
parser.add_argument('--valprop', type=float, help='proportion of training data to use for validation (default: 0.15)',
                    default=0.15)
parser.add_argument('--gpu', type=int, help='which GPU to run on (default: 0)', default=0)
parser.add_argument('--result_folder', type=str, help='where to save the results (default: ./results/)',
                    default='./results/')
parser.add_argument('--data_folder', type=str, help='where to find/put the data (default: ../../data/)',
                    default='../../data/')
parser.add_argument('--previous_result_folder', type=str, default=None,
                    help='previous result folder from which to continue (default: None)')
parser.add_argument('--method', type=str, help='model to use (default: DUN_wd)',
                    default='DUN_wd')
parser.add_argument('--network', type=str,
                    help='model type when using DUNs (other methods use ResNet) (default: ResNet)',
                    default='ResNet', choices=['ResNet', 'MLP'])
parser.add_argument('--width', type=int, help='width of the hidden units (default: 100)', default=100)
parser.add_argument('--batch_size', type=int, help='training chunk size (default: 128)', default=128)
parser.add_argument('--early_stop', type=int, default=None,
                    help='number of iterations of no improvement before early stopping (default: None)')
parser.add_argument('--num_workers', type=int, default=1,
                    help='number of parallel (threaded) workers to run (default: 1)')


def main(args):
    extra_string = ''

    if args.dataset == 'flights':
        if args.n_split == 0:
            extra_string += '_2M'
        elif args.n_split == 1:
            extra_string += '_800k'
        else:
            raise Exception('Only Valid values for flight splits are 0 (2M) or 1 (800k)')
        extra_string += '_valprop_' + str(args.valprop)

    elif args.dataset in ['boston', 'concrete', 'energy', 'power', 'wine', 'yacht', 'kin8nm', 'naval', 'protein',
                          'boston_gap', 'concrete_gap', 'energy_gap', 'power_gap', 'wine_gap', 'yacht_gap',
                          'kin8nm_gap', 'naval_gap', 'protein_gap']:
        extra_string += '_split_' + str(args.n_split)
        extra_string += '_valprop_' + str(args.valprop)

    working_dir = args.result_folder + '/' + args.dataset + extra_string + '/' + args.method +\
        ('-' + args.network if args.network != "ResNet" else '') + '/' + str(args.width) + '/' + str(args.batch_size) +\
        '/' + args.run_id
    print("WORKING DIR")
    print(working_dir)

    # Create data dir if necessary
    if not os.path.exists(args.data_folder):
        mkdir(args.data_folder)

    # Every process has to lookup the hostname
    host = hpns.nic_name_to_host(args.nic_name)

    result_logger = hpres.json_result_logger(directory=working_dir, overwrite=False)

    # Start a nameserver:
    NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=working_dir)
    ns_host, ns_port = NS.start()

    workers = []
    for i in range(args.num_workers):
        print("CREATING WORKER:", i)
        if args.dataset == 'spirals':
            worker_class = create_SpiralsWorker(args.method, args.network, args.width, args.batch_size)
            worker = worker_class(early_stop=args.early_stop, run_id=args.run_id, host=host, nameserver=ns_host,
                                  nameserver_port=ns_port, timeout=600, id=i)
        elif args.dataset == 'flights':
            worker_class = create_FlightWorker(args.method, args.network, args.width, args.batch_size)
            worker = worker_class(base_dir=args.data_folder, prop_val=args.valprop, k800=(args.n_split == 1),
                                  early_stop=args.early_stop, run_id=args.run_id, host=host,
                                  nameserver=ns_host, nameserver_port=ns_port, timeout=600, id=i)
        elif args.dataset in ['boston', 'concrete', 'energy', 'power', 'wine', 'yacht', 'kin8nm', 'naval', 'protein',
                              'boston_gap', 'concrete_gap', 'energy_gap', 'power_gap', 'wine_gap', 'yacht_gap',
                              'kin8nm_gap', 'naval_gap', 'protein_gap']:
            worker_class = create_UCIWorker(args.method, args.network, args.width, args.batch_size)
            worker = worker_class(dname=args.dataset, base_dir=args.data_folder, prop_val=args.valprop,
                                  n_split=args.n_split, early_stop=args.early_stop, run_id=args.run_id,
                                  host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=600, id=i)
        else:
            raise ValueError('Dataset not implemented yet!')

        worker.run(background=True)
        workers.append(worker)

    n_iterations = args.n_iterations
    previous_run = None
    if args.previous_result_folder is not None:
        try:
            previous_run = hpres.logged_results_to_HBS_result(args.previous_result_folder)
        except Exception as e:
            print(e)

    # Run an optimizer
    bohb = BOHB(
        configspace=worker.get_configspace(),
        run_id=args.run_id,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        result_logger=result_logger,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        previous_result=previous_run,
    )

    res = bohb.run(n_iterations=n_iterations, min_n_workers=args.num_workers)

    # store results
    with open(os.path.join(working_dir, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    # shutdown
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    all_runs = res.get_all_runs()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.' % (sum([r.budget for r in all_runs]) /
                                                                           args.max_budget))
    print('The run took  %.1f seconds to complete.' % (all_runs[-1].time_stamps['finished'] -
                                                       all_runs[0].time_stamps['started']))

if __name__ == "__main__":
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu

    main(args)
