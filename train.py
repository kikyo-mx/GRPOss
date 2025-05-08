import torch
import argparse
import torch.nn
import numpy as np
from dataloader import USData
import os
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import pandas as pd
from stocktrendr import stock_trendr
from torch.utils.data import DataLoader
from models.GRPO import ReplayMemory, PolicyNet, GRPOLoss
from models.stockENV import StockENV, hyedge_list, get_mask

seed = 123456789
np.random.seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='box')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--epochs', type=int, default=100, help='training epochs (default: 150)')
parser.add_argument('--device', type=str, default='cuda:1', help='running hardware (default: GPU)')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-4)')
parser.add_argument('--year', type=int, default=2022, help='data year (default: 2013)')
parser.add_argument('--gamma', type=float, default=0.99, help='(default: 0.9)')
parser.add_argument('--bs', type=int, default=32, help='batch size(default: 8192)')
parser.add_argument('--ms', type=int, default=5000, help='memory size(default: 40000)')
parser.add_argument('--weight', type=int, default=5, help='stocktrendr weight(default: 50)')
parser.add_argument('--rti', type=int, default=10, help='replace target model iter(default: 50)')
parser.add_argument('--st', type=tuple, default=(0.1, 1), help='stocktrendr slop and res(default: 0.05, 2)')
parser.add_argument('--e', type=float, default=0.5, help='epsilon (default: 0.4)')
parser.add_argument('--sector', type=str, default='C', help='sector (default:''A'')')
parser.add_argument('--g', type=str, default=100, help='groups num (default:''100'')')
parser.add_argument('--topn', type=str, default=5, help='top N stocks (default:''5'')')
parser.add_argument('--klw', type=float, default=0.1, help='kl_weight (default:''0.1'')')

args = parser.parse_args()

# pre_config
use_PreAttn = True
use_HidAttn = True
log_path = '/home/kikyo/code/log/'
write_log = 1
en_model = 'GRU'
if write_log:
    writer = SummaryWriter(log_path)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_path])
    url = tb.launch()

pd_col = ['train_loss', 'val_loss', 'val_roi', 'val_sharp', 'val_rank']
pd_output = pd.DataFrame(columns=pd_col)

# os.remove(log_path + os.listdir(log_path)[0])
market_name = 'NASDAQ'
root_path = '/home/kikyo/data/qt/'
data_path = os.path.join(root_path, market_name)
inci_mat = np.load('hg_new_' + market_name + '.npy')
hyedge = hyedge_list(inci_mat)
tickers = os.path.join(root_path, market_name + '_tickers_qualify_dr-0.98_min-5_smooth.csv')
tickers = np.genfromtxt(tickers,
                        dtype=str, delimiter='\t', skip_header=False)
PreConcept = torch.from_numpy(np.load('/home/kikyo/data/qt/PreConceptWeight_' + market_name + '.npy')).to(args.device)
hyp_input = torch.from_numpy(inci_mat).to(args.device)

# hyper parameters
seq_len = 8
rr_num = [0, 1, 2]
loss_weight = torch.ones(len(rr_num)).to(args.device)
flag = ['train', 'test', 'val']

GRPO = GRPOLoss(kl_weight=args.klw, device=args.device)

# load data
data_set = USData(root_path=data_path, market=market_name, tickers=tickers, seq_len=seq_len)
train_size = int(len(data_set) * 0.5)
validate_size = int(len(data_set) * 0.3)
test_size = len(data_set) - train_size - validate_size
train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(data_set,
                                                                              [train_size, validate_size, test_size])
trainLoader = DataLoader(train_dataset, shuffle=False, drop_last=True)
validateLoader = DataLoader(validate_dataset, shuffle=False, drop_last=True)
testLoader = DataLoader(test_dataset, shuffle=False, drop_last=True)

print(len(data_set), 'data_set')
print(len(trainLoader), 'train_loader')

train_env = StockENV(trainLoader, device=args.device)
test_env = StockENV(testLoader, device=args.device)

# load model
policy_model = PolicyNet(5, 1).to(args.device)
old_model = PolicyNet(5, 1).to(args.device)
ref_model = PolicyNet(5, 1).to(args.device)
optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr)
MSE = torch.nn.MSELoss().to(args.device)
BCE = torch.nn.BCELoss().to(args.device)
sm = torch.nn.Softmax(dim=1).to(args.device)

# for name, param in policy_model.named_parameters():
#     if 'weight' in name:
#         torch.nn.init.normal_(param, mean=0, std=0)
#     if 'bias' in name:
#         torch.nn.init.constant_(param, val=0)
# train
def main():
    loop1 = tqdm(range(args.epochs), total=args.epochs, ncols=100)
    learn_step_counter = 0
    replace_target_iter = args.rti
    for epoch in loop1:
        start_time = time.time()
        train_loss = []
        policy_model.train()
        ref_model.load_state_dict(policy_model.state_dict())
        for param in ref_model.parameters():
            param.requires_grad = False
        for i, (stocks_data, stocks_close, stocks_open, stocks_close_trend, mask_batch, macro_data) in enumerate(trainLoader):
            stocks_data = stocks_data.squeeze(0)
            if learn_step_counter % replace_target_iter == 0:
                old_model.load_state_dict(policy_model.state_dict())
                for param in old_model.parameters():
                    param.requires_grad = False

            # stockstrendr
            hyedge_weight = torch.ones(len(hyedge)).to(args.device)
            # stocks_close_trend = stocks_close_trend.squeeze(0).numpy()
            # for j in range(len(hyedge)):
            #     cur_obox = stock_trendr(np.mean(stocks_close_trend[hyedge[j]], axis=0), 0, 30, args.st)
            #     if cur_obox and cur_obox[-1][1] >= 29:
            #         hyedge_weight[j] = args.weight

            # GRPO Learning
            output = old_model(stocks_data.to(args.device), hyp_input, hyedge_weight)
            mask = mask_batch.to(args.device).bool().squeeze(0)
            output = output * mask # (1026, 1) NASDAQ
            aciton_groups = train_env.tack_action(output, args.topn, args.g)
            reward, advantages = train_env.step(aciton_groups, i)

            old_logprobs = torch.log(output[aciton_groups])
            new_logprobs = torch.log(policy_model(stocks_data.to(args.device), hyp_input, hyedge_weight)[aciton_groups])
            ref_logprobs = torch.log(ref_model(stocks_data.to(args.device), hyp_input, hyedge_weight)[aciton_groups])

            groups_mask = get_mask(aciton_groups, stocks_close_trend.squeeze(0).numpy(), args.st, args.weight).to(args.device)

            tra_loss = GRPO.compute_loss(old_logprobs, new_logprobs, ref_logprobs, advantages, groups_mask).mean()

            optimizer.zero_grad()
            tra_loss.backward()
            optimizer.step()
            train_loss.append(tra_loss.cpu().detach().numpy())

            learn_step_counter += 1
        train_loss = np.average(train_loss)

        if write_log:
            writer.add_scalar('train_loss', train_loss, global_step=epoch)

        loop1.set_description(f'Epoch [{epoch + 1}/{args.epochs}] Train Performance')
        loop1.set_postfix(tra_loss=train_loss)

    output_name = ''
    # if compare:
    #     pd_output.to_csv('./output/' + market_name + '_' + en_model + '_' + output_name + '_output.csv')
    # else:
    #     pd_output.to_csv('./output/' + market_name + '_' + output_name + '_output.csv')

if __name__ == '__main__':
    main()
