import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import argparse
import random
from loss import ContrastiveLoss
from dataloader import load_data
from causal_module import CausalDebiasedMultiViewClustering, CausalContrastiveLoss, ViewInvarianceLoss


#Yale
#ALOI
#Animal
#OutdoorScene
#COIL20
#EYaleB
#ORL


Dataname = 'Yale'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003, type=float)
parser.add_argument("--weight_decay", default=0., type=float)
parser.add_argument("--pre_epochs", default=200, type=int)
parser.add_argument("--con_epochs", default=50, type=int)
parser.add_argument("--feature_dim", default=64, type=int)
parser.add_argument("--high_feature_dim", default=20, type=int)
parser.add_argument("--temperature", default=1, type=float)
parser.add_argument("--causal_weight", default=0.5, type=float, help="Overall weight for causal structure (Legacy, kept for backward compatibility)")
parser.add_argument("--alpha", default=1.0, type=float, help="Weight for Disentanglement loss (Lrec + Lalign)")
parser.add_argument("--beta", default=1.0, type=float, help="Weight for Invariance loss (Linv)")
parser.add_argument("--no_counterfactual", action="store_true", help="Disable counterfactual augmentation")
parser.add_argument("--no_crm", action="store_true", help="Disable Causal Reweighting Module (CRM)")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "COIL20":
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "ALOI":
    # Medium dataset
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 100
    seed = 10

if args.dataset == "OutdoorScene":
    args.batch_size = 256
    args.learning_rate = 3e-4
    args.con_epochs = 50
    seed = 10

if args.dataset == "Animal":
    args.batch_size = 512 
    args.learning_rate = 1e-3
    args.con_epochs = 50 
    seed = 10

if args.dataset == "Yale":
    args.batch_size = 64
    args.learning_rate = 1e-3
    args.con_epochs = 200
    seed = 10

if args.dataset == "ORL":
    args.batch_size = 64
    args.learning_rate = 1e-3
    args.con_epochs = 200
    seed = 10

if args.dataset == "EYaleB":
    args.batch_size = 64
    args.learning_rate = 1e-3
    args.con_epochs = 100
    seed = 10


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

def compute_view_value(rs, H, view):
    N = H.shape[0]
    w = []
    global_sim = torch.matmul(H,H.t())
    for v in range(view):
        view_sim = torch.matmul(rs[v],rs[v].t())
        related_sim = torch.matmul(rs[v],H.t())
        w_v = (torch.sum(view_sim) + torch.sum(global_sim) - 2 * torch.sum(related_sim)) / (N*N)
        w.append(torch.exp(-w_v))
    w = torch.stack(w)
    w = w / torch.sum(w)
    return w.squeeze()


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        zs = []
        for v in range(view):
            zs.append(model.encoders[v](xs[v]))
    
        c_list, c_cf_list, z_rec_list, z_cf_list = causal_module(zs, return_counterfactuals=False)
        
        xrs = []
        for v in range(view):
            xrs.append(model.decoders[v](zs[v]))

        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        
        l_rec, l_inv, l_align = causal_contrastive_criterion(zs, c_list, c_cf_list, z_rec_list)

        causal_loss_term = args.alpha * (l_rec + l_align) + args.beta * l_inv
        
        loss = sum(loss_list) + causal_loss_term
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    avg_loss = tot_loss/len(data_loader)
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(avg_loss))
    return avg_loss

def contrastive_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()

    warmup_threshold = args.pre_epochs + int(0.4 * args.con_epochs)
    use_cf = (not args.no_counterfactual) and (epoch > warmup_threshold)
    
    if use_cf and epoch == warmup_threshold + 1:
        print(">>> [Causal Warm-up Finished] Style-shuffled Counterfactuals Enabled.")

    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()

        zs = []
        for v in range(view):
            zs.append(model.encoders[v](xs[v], apply_crm=use_cf))
        c_list, c_cf_list, z_rec_list, z_cf_list = causal_module(zs, return_counterfactuals=use_cf)
        
        rs = []
        for v in range(view):
            rs.append(F.normalize(model.common_information_module(zs[v]), dim=1))
        
        H = model.feature_fusion(zs, zs_gradient=True)

        xrs = []
        for v in range(view):
            xrs.append(model.decoders[v](zs[v]))
        
        loss_list = []
        with torch.no_grad():
            w = compute_view_value(rs, H, view)

        for v in range(view):
            loss_list.append(contrastiveloss(H, rs[v], w[v]))
            loss_list.append(mse(xs[v], xrs[v]))
        
        l_rec, l_inv, l_align = causal_contrastive_criterion(zs, c_list, c_cf_list, z_rec_list)
        
        causal_loss_term = args.alpha * (l_rec + l_align) + args.beta * l_inv
        
        loss = sum(loss_list) + causal_loss_term
        
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    avg_loss = tot_loss/len(data_loader)
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(avg_loss))
    return avg_loss


accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
for i in range(T):
    print("ROUND:{}".format(i+1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, device, use_crm=not args.no_crm)

    causal_module = CausalDebiasedMultiViewClustering(
        num_views=view, 
        feature_dim=args.feature_dim, 
        device=device
    )
    print(model)
    model = model.to(device)
    causal_module = causal_module.to(device)
    
    state = model.state_dict()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(causal_module.parameters()), 
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    contrastiveloss = ContrastiveLoss(args.batch_size, args.temperature, device).to(device)
    
    causal_contrastive_criterion = CausalContrastiveLoss(temperature=args.temperature).to(device)
    best_acc, best_nmi, best_pur = 0, 0, 0

    epoch = 1
    while epoch <= args.pre_epochs:
        loss = pretrain(epoch)
        epoch += 1

    while epoch <= args.pre_epochs + args.con_epochs:
        loss = contrastive_train(epoch)
        acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=epoch)

        if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
            torch.save(causal_module.state_dict(), './models/' + args.dataset + '_causal.pth')
        epoch += 1

    accs.append(best_acc)
    nmis.append(best_nmi)
    purs.append(best_pur)
    print('The best clustering performace: ACC = {:.4f} NMI = {:.4f} PUR={:.4f}'.format(best_acc, best_nmi, best_pur))
