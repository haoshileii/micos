import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from micos import micos
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
import xlrd
import xlwt
from xlutils.copy import copy
import matplotlib.pyplot as ply
def save_checkpoint_callback(
        save_every=1,
        unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback
if __name__ == '__main__':
    temperal_units = [5]
    temperal_num = 0
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("MultivariableTS")
    workbook.save("resultstestunit.xls")
    for temperal in temperal_units:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default='BasicMotions', help='The dataset name')
        parser.add_argument('--run_name', default='UEA',
                            help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
        parser.add_argument('--gpu', type=int, default=0,
                            help='The gpu no. used for training and inference')
        parser.add_argument('--batch-size', type=int, default=8, help='The batch size')
        parser.add_argument('--lr', type=int, default=0.0005, help='The learning rate')
        parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension')
        parser.add_argument('--max-train-length', type=int, default=30000,
                            help='')
        parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
        parser.add_argument('--epochs', type=int, default=500, help='The number of epochs')
        parser.add_argument('--save-every', type=int, default=None,
                            help='Save the checkpoint every <save_every> iterations/epochs')
        parser.add_argument('--seed', type=int, default=42, help='The random seed')
        parser.add_argument('--max-threads', type=int, default=8,
                            help='The maximum allowed number of threads used by this process')
        parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
        args = parser.parse_args()
        device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        config = dict(
            batch_size=args.batch_size,
            lr=args.lr,
            output_dims=args.repr_dims,
            max_train_length=args.max_train_length
        )
        if args.save_every is not None:
            unit = 'epoch' if args.epochs is not None else 'iter'
            config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)
        run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
        os.makedirs(run_dir, exist_ok=True)
        t = time.time()
        model = micos(
            input_dims=train_data.shape[-1],
            temporal_unit=temperal,
            device=device,
            **config
        )
        loss_log, p_epoch = model.fit(
            train_data,
            train_labels,
            n_epochs=args.epochs,
            n_iters=args.iters,
            verbose=True
        )
        model.save(f'{run_dir}/model.pkl')
        t = time.time() - t
        print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
        if args.eval:
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels,
                                                        eval_protocol='svm')
            pkl_save(f'{run_dir}/out.pkl', out)
            pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
            print('Evaluation result:', eval_res['acc'])
        readbook = xlrd.open_workbook("resultstestunit.xls")
        wb = copy(readbook)
        sh1 = wb.get_sheet(0)
        sh1.write(0, temperal_num, temperal)
        sh1.write(1, temperal_num, str(eval_res['acc']))
        wb.save('resultstestunit.xls')
        temperal_num = temperal_num + 1
        print("Finished.")
