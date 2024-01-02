import argparse
import os
import torch

from exp.exp_informer import Exp_Informer

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

#网络结构，default更改，删掉“required=True”可以debug
# parser.add_argument('--model', type=str, required=True, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')

#读的数据，类型、路径\文件名
# parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
# parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
# parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')  

# parser.add_argument('--data', type=str, required=True, default='WTH', help='data')
parser.add_argument('--data', type=str, default='WTH', help='data')
parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='WTH.csv', help='data file')  

#预测数据类型，单变量多变量  
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

#哪一列是标签，ground truth
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')

#数据中的时间的单位
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

#模型最后保留位置
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

#encoder输入序列长度（可自定义）
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')

#decoder输入长度（可自定义）
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')

#预测序列长度（可自定义）
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

#编码器解码器输入输出维度
parser.add_argument('--enc_in', type=int, default=12, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=12, help='decoder input size')

#输出size
parser.add_argument('--c_out', type=int, default=12, help='output size')

#特征维度
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')

#多头注意力头数
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')

#encoder/decoder layer数量
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')

#encoder block数
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')

#fc层数
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')

#Q采用因子
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')

#填充
parser.add_argument('--padding', type=int, default=0, help='padding type')

#是否下采样操作pooling，减小序列长度？减半
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)

#dropout
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

#注意力机制
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')

#嵌入
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')

#激活函数
parser.add_argument('--activation', type=str, default='gelu',help='activation')

#编码器解码器注意力
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

#预测
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

#informer框架里没有
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)

#读数据
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')

#Windows用户是0？
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

#训练，iteration是epoch的次数
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

#早停策略
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')

parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

#？
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')

#分布式，适合linux？
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

#翻转输出数据，干啥的？
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

#gpu，默认用，一个gpu就是0
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

#多个gpu
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)

#如果是分布式，用几个显卡
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

#用Exp_Informer取数据 阿萨德
Exp = Exp_Informer

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, ii)

    exp = Exp(args) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    #调到train函数
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()




