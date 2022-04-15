import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSEncoder
from models.lossestriples import hierarchical_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
from dataaugmentations import data_augmentation
import math
import random
# import matplotlib
# matplotlib.rcParams['backend'] = 'SVG'
# import matplotlib.pyplot as ply
# from visdom import Visdom
# vis=Visdom(env="heat_map")



class TS2Vec:
    '''The TS2Vec model'''
    def __init__(#一些参数和默认值
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        #初始化自身的一些参数
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims,  hidden_dims=hidden_dims, depth=depth).to(self.device)
        #权重参数平均模型
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
    # 训练表示模型 CBF的n_epochs=None, n_iters=None
    def fit(self, train_data, train_labels, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the TS2Vec model.2
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs（所有训练集反复训练的次数）. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations（反向传播次数）. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        # 断言 当后面的断言为真时，程序继续往下执行，否则程序抛出异常
        assert train_data.ndim == 3
        #如果迭代次数和反向传播次数没有设置，就默认200或者600
        #CBF:train_data:(30,128,1)
        #print("//////////////////////////////////")
        #print(train_data.size)   3840
        #print("//////////////////////////////////")
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
        # print(train_data.shape[1])
        # print(self.max_train_length)
        #CBF:n_iters=200

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            # print(sections)如果是规则时间序列该值为0，不再往下执行
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=-1)
                # #临时代码，只针对EM
                # train_data = torch.from_numpy(train_data)
                # train_labels = torch.from_numpy(train_labels)
                # train_data_labels = {'samples':train_data,'labels':train_labels}
                # torch.save(train_data_labels, 'datasets/BTS/EigenWorms/test.pt')
        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        #print("----------------")
        #print(temporal_missing[127])

        #print(temporal_missing[-1])
        #print("----------------")
        # 规则时间序列中上面两者都等于false,所以下面的if语句不执行
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
        # AA= np.isnan(train_data)
        # print(AA.shape)
        # BB = ~np.isnan(train_data)
        # print(BB.shape)
        # CC = ~np.isnan(train_data).all(axis=2)
        # print(CC.shape)
        # DD = ~np.isnan(train_data).all(axis=2).all(axis=1)
        # print(DD.shape)
        #删除数组中所有含有缺失值的行
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        # 将数据集转换为张量格式并压缩
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float), torch.from_numpy(train_labels).to(torch.float))
        #将数据变成若干batch的形式，drop_last=True：最后一个bacth数量不够会被扔掉  shuffle：在每一个epoch开始时候会对数据进行重新排序
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        #定义优化方法
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        loss_log = []
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            cum_loss = 0
            n_epoch_iters = 0
            interrupted = False
            for id, batch in enumerate(train_loader,1):
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                x = batch[0]
                sup_labels = batch[1]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                ########如果采用转换增强
                # nm_train = random.randint(0, 2)
                # if nm_train != 0:
                #     x, sup_labels = data_augmentation(x, sup_labels, nm_train)
                ########如果采用转换增强
                x = x.to(self.device)
                ts_l = x.size(1)
                crop_l = np.random.randint(2 ** (self.temporal_unit + 1), high=ts_l+1)
                print(crop_l)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                nu_trans = np.random.randint(3, 4)
                #print(nu_trans)
                if nu_trans == 2:
                    crop_eleft = np.random.randint(crop_left + 1)
                    crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                    crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                    optimizer.zero_grad()
                    out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                    out1 = out1[:, -crop_l:]
                    out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                    out2 = out2[:, :crop_l]
                    out = torch.cat([out1, out2], 0)
                    sup_labels = sup_labels.repeat(2)
                else:
                    crop_eleft_1 = np.random.randint(crop_left + 1)
                    crop_eright_1 = np.random.randint(low=crop_right, high=ts_l + 1)
                    crop_eleft_2 = np.random.randint(crop_eleft_1 + 1)
                    crop_eright_2 = np.random.randint(low=crop_eright_1, high=ts_l + 1)
                    crop_offset = np.random.randint(low=-crop_eleft_2, high=ts_l - crop_eright_2 + 1, size=x.size(0))
                    optimizer.zero_grad()
                    out1 = self._net(take_per_row(x, crop_offset + crop_eleft_2, crop_right - crop_eleft_2))
                    out1 = out1[:, -crop_l:]
                    out2 = self._net(take_per_row(x, crop_offset + crop_eleft_1, crop_eright_1 - crop_eleft_1))
                    out2 = out2[:, (crop_left-crop_eleft_1):(crop_right-crop_eleft_1)]
                    out3 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright_2 - crop_left))
                    out3 = out3[:, :crop_l]
                    out = torch.cat([out1, out2, out3], 0)
                    sup_labels = sup_labels.repeat(3)
                # loss = hierarchical_contrastive_loss(
                #     out,
                #     sup_labels,
                #     temporal_unit=self.temporal_unit
                # )
                ########################普通增强
                loss = hierarchical_contrastive_loss(
                    nu_trans,
                    out,
                    sup_labels,
                    temporal_unit=self.temporal_unit
                )
                ########################普通增强




                ##################if not augmentation
                # loss = hierarchical_contrastive_loss(
                #     out1,
                #     out2,
                #     sup_labels,
                #     temporal_unit=self.temporal_unit
                # )
                ##################if not augmentation
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters#每一次的epoch输出的是一个epoch里iteration的平均损失
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

        return loss_log, self.n_epochs
    
    def _eval_with_pooling(self, x, draw_i, x_label, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        #画表示出来的图形, 两种类别
        # if draw_i == 1:
        #     print(x_label)
        #     print('第一类ENCODER')
        #     out_draw1 = out.to(device=x.device)
        #     out_draw2 = out_draw1.cuda().data.cpu().numpy()
        #     out_draw2_1 = out_draw2[0]
        #     out_draw2_2 = out_draw2[1]
        #     out_draw2_3 = out_draw2[2]
        #     out_draw2_4 = out_draw2[3]
        #     out_draw2_5 = out_draw2[4]
        #     out_draw2_6 = out_draw2[5]
        #     out_draw2_7 = out_draw2[6]
        #     out_draw2_8 = out_draw2[7]
        #
        #     np.savetxt('Epilepsy1.txt', out_draw2_1)
        #     np.savetxt('Epilepsy2.txt', out_draw2_2)
        #     np.savetxt('Epilepsy3.txt', out_draw2_3)
        #     np.savetxt('Epilepsy4.txt', out_draw2_4)
        #     np.savetxt('Epilepsy5.txt', out_draw2_5)
        #     np.savetxt('Epilepsy6.txt', out_draw2_6)
        #     np.savetxt('Epilepsy7.txt', out_draw2_7)
        #     np.savetxt('Epilepsy8.txt', out_draw2_8)
        # if draw_i == 10:
        #     print(x_label)
        #     print('第二类ENCODER')
        #     out2_draw1 = out.to(device=x.device)
        #     out2_draw2 = out2_draw1.cuda().data.cpu().numpy()
        #     out2_draw2_1 = out2_draw2[0]
        #     out2_draw2_2 = out2_draw2[1]
        #     out2_draw2_3 = out2_draw2[2]
        #     out2_draw2_4 = out2_draw2[3]
        #     out2_draw2_5 = out2_draw2[4]
        #     out2_draw2_6 = out2_draw2[5]
        #     out2_draw2_7 = out2_draw2[6]
        #     out2_draw2_8 = out2_draw2[7]
        #
        #     np.savetxt('out2Epilepsy1.txt', out2_draw2_1)
        #     np.savetxt('out2Epilepsy2.txt', out2_draw2_2)
        #     np.savetxt('out2Epilepsy3.txt', out2_draw2_3)
        #     np.savetxt('out2Epilepsy4.txt', out2_draw2_4)
        #     np.savetxt('out2Epilepsy5.txt', out2_draw2_5)
        #     np.savetxt('out2Epilepsy6.txt', out2_draw2_6)
        #     np.savetxt('out2Epilepsy7.txt', out2_draw2_7)
        #     np.savetxt('out2Epilepsy8.txt', out2_draw2_8)

        if draw_i == 1:
            print(x_label)
            print('第0类ENCODER')
            out0_draw1 = out.to(device=x.device)
            out0_draw2 = out0_draw1.cuda().data.cpu().numpy()
            out0_draw2_1 = out0_draw2[0]
            out0_draw2_2 = out0_draw2[1]

            np.savetxt('EthanolConcentration01.txt', out0_draw2_1)#第0类的第1个样本
            np.savetxt('EthanolConcentration02.txt', out0_draw2_2)#第0类的第2个样本

        if draw_i == 5:
            print(x_label)
            print('第1类ENCODER')
            out1_draw1 = out.to(device=x.device)
            out1_draw2 = out1_draw1.cuda().data.cpu().numpy()
            out1_draw2_1 = out1_draw2[0]
            out1_draw2_2 = out1_draw2[1]

            np.savetxt('EthanolConcentration11.txt', out1_draw2_1)
            np.savetxt('EthanolConcentration12.txt', out1_draw2_2)
        if draw_i == 10:
            print(x_label)
            print('第2类ENCODER')
            out2_draw1 = out.to(device=x.device)
            out2_draw2 = out2_draw1.cuda().data.cpu().numpy()
            out2_draw2_1 = out2_draw2[0]
            out2_draw2_2 = out2_draw2[1]

            np.savetxt('EthanolConcentration21.txt', out2_draw2_1)
            np.savetxt('EthanolConcentration22.txt', out2_draw2_2)
        if draw_i == 16:
            print(x_label)
            print('第3类ENCODER')
            out3_draw1 = out.to(device=x.device)
            out3_draw2 = out3_draw1.cuda().data.cpu().numpy()
            out3_draw2_1 = out3_draw2[0]
            out3_draw2_2 = out3_draw2[1]

            np.savetxt('EthanolConcentration31.txt', out3_draw2_1)
            np.savetxt('EthanolConcentration32.txt', out3_draw2_2)

        # plot_x = out_draw2[1]
        # vis.heatmap(
        #     X=plot_x,
        #     opts=dict(
        #         colormap='Electric',
        #     )
        # )
        # plot_x = list(range(1, out_draw2.shape[1] + 1))
        # fig, ax = ply.subplots()
        # for i in range(len(plot_y)):
        #     ax.plot(np.array(plot_x), plot_y[i])
        #
        # # ply.legend()
        # ax.tick_params(bottom=True, top=False, left=True, right=False)  # 移除全部刻度线
        # ply.xticks([])
        # ply.yticks([])
        # ply.savefig('results/out_draw1.svg', format='svg')

        #画表示出来的图形
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
        #输出池化之后的表示
        # out_squeeze = out.to(device=x.device)
        # if draw_i == 1:
        #     out_squeeze = out_squeeze.squeeze(1)
        #     out_squeeze_out = out_squeeze.cuda().data.cpu().numpy()
        #     out_squeeze_out1 = out_squeeze_out[0]
        #     out_squeeze_out2 = out_squeeze_out[1]
        #     np.savetxt('Epilepsyafterpooling01.txt', out_squeeze_out1)
        #     np.savetxt('Epilepsyafterpooling02.txt', out_squeeze_out2)
        # if draw_i == 5:
        #     out_squeeze = out_squeeze.squeeze(1)
        #     out_squeeze_out = out_squeeze.cuda().data.cpu().numpy()
        #     out_squeeze_out1 = out_squeeze_out[0]
        #     out_squeeze_out2 = out_squeeze_out[1]
        #     np.savetxt('Epilepsyafterpooling11.txt', out_squeeze_out1)
        #     np.savetxt('Epilepsyafterpooling12.txt', out_squeeze_out2)
        # if draw_i == 10:
        #     out_squeeze = out_squeeze.squeeze(1)
        #     out_squeeze_out = out_squeeze.cuda().data.cpu().numpy()
        #     out_squeeze_out1 = out_squeeze_out[0]
        #     out_squeeze_out2 = out_squeeze_out[1]
        #     np.savetxt('Epilepsyafterpooling21.txt', out_squeeze_out1)
        #     np.savetxt('Epilepsyafterpooling22.txt', out_squeeze_out2)
        # if draw_i == 16:
        #     out_squeeze = out_squeeze.squeeze(1)
        #     out_squeeze_out = out_squeeze.cuda().data.cpu().numpy()
        #     out_squeeze_out1 = out_squeeze_out[0]
        #     out_squeeze_out2 = out_squeeze_out[1]
        #     np.savetxt('Epilepsyafterpooling31.txt', out_squeeze_out1)
        #     np.savetxt('Epilepsyafterpooling32.txt', out_squeeze_out2)
        #输出池化之后的表示
        return out.cpu()
    
    def encode(self, data, data_labels, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        #画可解释性图
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float), torch.from_numpy(data_labels).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        #画可解释性图
        # dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        # loader = DataLoader(dataset, batch_size=batch_size)
        #验证集上面不需要进行反向传播来更新参数了，只需要得到网络模型的输出结果，torch.no_grad()就是起这个作用
        with torch.no_grad():
            output = []
            draw_i = 0 #the number of BATCH
            for batch in loader:
                #batch此时是一个3维张量
                x = batch[0]
                x_label = batch[1]
                #print("ppppppppppppppppppppppppp")
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(  
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, draw_i, x_label, mask, encoding_window=encoding_window)
                    draw_i = draw_i+1
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)#进行维度压缩
                        
                output.append(out)#在末尾添加新张量
                
            output = torch.cat(output, dim=0)#张量拼接
            
        self.net.train(org_training)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
    
