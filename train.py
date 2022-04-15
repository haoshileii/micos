import torch
import numpy as np
import argparse
import os
import sys

import time
import datetime
from ts2vec import TS2Vec
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
    assert unit in ('epoch', 'iter')  # 断言，值为false则中断程序执行

    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')  # 将模型的一些暂时变量或者需要暂存的字符串等保存起来

    return callback  # 执行该函数，返回空值


if __name__ == '__main__':  # 后面的代码只有当本脚本直接被执行时才会运行，如果该脚本是被其他文件调用，则后面的代码不会执行。
#     batchsize_name = [32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112,	120, 128, 136, 144,	152, 160, 168, 176,	184, 192, 200, 208,	216, 224, 232, 240,	248, 256, 264, 272,	280, 288
#
# ]
    #batchsize_name = [6, 7, 9, 10, 11, 12, 13, 14, 15]
    #batchsize_name = [8]
    #batchsize_num = 0
    #temperal_units = [0, 1, 2, 3, 4, 5]
    temperal_units = [1]
    temperal_num = 0
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("MultivariableTS")
    workbook.save("resultstestunit.xls")
    for temperal in temperal_units:
        parser = argparse.ArgumentParser()
        parser.add_argument('dataset', help='The dataset name')
        parser.add_argument('run_name',
                            help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
        parser.add_argument('--archive', type=str, required=True,
                            help='The archive name that the dataset belongs to. This can be set to UCR, UEA, forecast_csv, or forecast_csv_univar')
        parser.add_argument('--gpu', type=int, default=0,
                            help='The gpu no. used for training and inference (defaults to 0)')
        parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
        parser.add_argument('--lr', type=int, default=0.0005, help='The learning rate (defaults to 0.001)')
        parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
        parser.add_argument('--max-train-length', type=int, default=3000,
                            help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
        parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
        parser.add_argument('--epochs', type=int, default=1, help='The number of epochs')
        parser.add_argument('--save-every', type=int, default=None,
                            help='Save the checkpoint every <save_every> iterations/epochs')
        # 随机种子：
        parser.add_argument('--seed', type=int, default=None, help='The random seed')
        # 此进程使用的最大允许线程数
        parser.add_argument('--max-threads', type=int, default=None,
                            help='The maximum allowed number of threads used by this process')
        parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
        parser.add_argument('--irregular', type=float, default=0, help='The ratio of missing observations (defaults to 0)')
        args = parser.parse_args()  # 解析参数
        #args.batch_size = batchsizes
        print("Dataset:", args.dataset)
        print("Arguments:", str(args))
        # 这个函数的作用？
        device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

        if args.archive == 'UCR':
            task_type = 'classification'
            # CBF(30,128,1)(30,)(900,128,1)(900,)
            train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
        elif args.archive == 'UEA':
            task_type = 'classification'
            # FM(316,50,28)(316,)(100,50,28)(100,)
            train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        elif args.archive == 'BTS':
            task_type = 'classification'
            # FM(316,50,28)(316,)(100,50,28)(100,)
            train_data, train_labels, test_data, test_labels = datautils.load_BTS(args.dataset)
        elif args.archive == 'forecast_csv':
            task_type = 'forecasting'
            data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(
                args.dataset)
            train_data = data[:, train_slice]
        elif args.archive == 'forecast_csv_univar':
            task_type = 'forecasting'
            data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(
                args.dataset, univar=True)
            train_data = data[:, train_slice]
        elif args.archive == 'forecast_npy':
            task_type = 'forecasting'
            data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(
                args.dataset)
            train_data = data[:, train_slice]
        elif args.archive == 'forecast_npy_univar':
            task_type = 'forecasting'
            data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(
                args.dataset, univar=True)
            train_data = data[:, train_slice]
        else:
            raise ValueError(f"Archive type {args.archive} is not supported.")
        #     args.irregular=0#正常数据集不执行该if
        if args.irregular > 0:
            if task_type == 'classification':
                train_data = data_dropout(train_data, args.irregular)
                test_data = data_dropout(test_data, args.irregular)
            else:
                raise ValueError(f"Task type {task_type} is not supported when irregular is positive.")
        # 定义一个字典
        config = dict(
            batch_size=args.batch_size,  # default=8
            lr=args.lr,  # default=0.001
            output_dims=args.repr_dims,  # 应该指的是每一个时间戳的表示维度320
            max_train_length=args.max_train_length  # 允许输入的训练集的最大长度default=3000
        )

        if args.save_every is not None:
            unit = 'epoch' if args.epochs is not None else 'iter'
            config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

        run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
        os.makedirs(run_dir, exist_ok=True)
        # 返回当前的时间戳
        t = time.time()
        # train_data.shape[-1]:-1表示获取的是倒数第一个维度，这里指的是时间序列是几变量的
        # 调用表示模型并部分初始化
        # 建立一个神经网络模型对象
        # 对CBF input_dims=1；
        # 对FM，input_dims=28
        model = TS2Vec(
            input_dims=train_data.shape[-1],
            temporal_unit=temperal,
            device=device,
            **config
        )
        # 训练模型并得
        # 到最终的模型 输入参数args.epochs的default=none  args.iters=none verbose:每次epochs之后打印loss
        loss_log, p_epoch = model.fit(
            train_data,
            train_labels,
            n_epochs=args.epochs,
            n_iters=args.iters,
            verbose=True
        )

        model.save(f'{run_dir}/model.pkl')
        font2 = {'family': 'Times New Roman',

                 'weight': 'normal',

                 'size': 30,

                 }
        plot_x = list(range(1, p_epoch + 1))
        plot_y = loss_log
        ply.plot(np.array(plot_x), np.array(plot_y), linewidth=2)
        ply.title("PEMS-SF", font2)
        ply.xlabel("Epoch", font2)
        ply.ylabel("Loss", font2)
        #ply.ylim(0, 3)
        ply.grid(True)
        ply.grid(color='lightgray', linestyle = '--')
        ply.grid(alpha=1)
        ply.xticks(fontproperties='Times New Roman', size=22)
        ply.yticks(fontproperties='Times New Roman', size=22)
        ply.tick_params(axis="both", colors='black', labelsize=22)
        ax = ply.gca()  # 获取边框
        ax.spines['top'].set_color('black')  # 设置上‘脊梁’为黑色
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ply.savefig('results/PEMS-SFloss.svg')
        ply.show()
        #print('tu')
        # 模型的训练时间
        t = time.time() - t
        print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

        # eval为true
        if args.eval:
            if task_type == 'classification':
                out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels,
                                                        eval_protocol='svm')
            elif task_type == 'forecasting':
                out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens,
                                                    n_covariate_cols)
            else:
                assert False
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
