"""
train_gw_search.py — 引力波信号搜寻: 训练 + 评估
================================================

用法:
  # Phase 1: 跑通 BBH baseline (原版 CNN)
  python train_gw_search.py --phase baseline --epochs 100

  # Phase 2: BNS + ResNet
  python train_gw_search.py --phase bns_resnet --epochs 100

  # Phase 2 轻量版 (调试用)
  python train_gw_search.py --phase bns_resnet --epochs 30 --model small

依赖:
  pip install torch numpy matplotlib scikit-learn tqdm lalsuite
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import cycle
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score


# ============================================================
# 参数解析
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(description='GW signal search: BBH baseline / BNS+ResNet')
    parser.add_argument('--phase', type=str, default='baseline',
                        choices=['baseline', 'bns_resnet'],
                        help='baseline = 原版 BBH+CNN; bns_resnet = BNS+ResNet')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.003, help='学习率')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nsample', type=int, default=100, help='每 epoch 样本数')
    parser.add_argument('--pool_size', type=int, default=5000,
                        help='预生成数据池大小 (越大越好, 受限于内存)')
    parser.add_argument('--snr_train', type=float, default=20.0, help='训练 SNR')
    parser.add_argument('--model', type=str, default='full',
                        choices=['full', 'small'],
                        help='ResNet 版本: full=ResNet18, small=轻量版')
    parser.add_argument('--device', type=str, default='auto',
                        help='cuda / cpu / auto')
    parser.add_argument('--outdir', type=str, default=None,
                        help='输出目录 (默认根据 phase 自动选)')
    return parser.parse_args()


# ============================================================
# 数据集类 (兼容 BBH 和 BNS)
# ============================================================
class GWDataset(Dataset):
    """
    预生成数据池 + 每 epoch 随机抽取

    策略: 一次性调 lalsimulation 生成 pool_size 个样本存到内存,
    训练时每 epoch 从池子里随机抽 nsample_perepoch 个.
    彻底消除 CPU 波形生成的瓶颈.

    Parameters
    ----------
    source : str
        'bbh' 或 'bns'
    pool_size : int
        预生成的总样本数. 越大越好, 受限于内存.
        每个样本 ~2 探测器 × 16384 float64 ≈ 256 KB
        pool_size=5000 → ~1.3 GB, pool_size=10000 → ~2.5 GB
    nsample_perepoch : int
        每 epoch 从池子里抽多少个样本 (抽取后 shuffle)
    reshape_for_resnet : bool
        True → (ndet, N/2) for ResNet1D;  False → (1, ndet, N) for MyNet
    """

    def __init__(self, source='bbh', fs=8192, T=1, snr=20,
                 detectors=['H1', 'L1'],
                 pool_size=5000, nsample_perepoch=100,
                 Nnoise=25, reshape_for_resnet=False, verbose=True):
        if verbose:
            print(f'GPU available? {torch.cuda.is_available()}')
        self.source = source
        self.fs = fs
        self.T = T
        safe = 2
        self.T_obs = T * safe
        self.detectors = detectors
        self.snr = snr
        self.Nnoise = Nnoise
        self.reshape_for_resnet = reshape_for_resnet
        self.nsample_perepoch = nsample_perepoch

        # ========== 一次性预生成数据池 ==========
        if verbose:
            print(f'Pre-generating {pool_size} samples ({source.upper()}, SNR={snr})...')
        import time as _time
        t0 = _time.time()

        if source == 'bbh':
            from data_prep_bbh import sim_data
            ts, par = sim_data(fs, self.T_obs, snr, detectors,
                               Nnoise, size=pool_size, mdist='metric',
                               beta=[0.75, 0.95], verbose=False)
        elif source == 'bns':
            from data_prep_bns import sim_data_bns
            ts, par = sim_data_bns(fs, self.T_obs, snr, detectors,
                                   Nnoise, size=pool_size, beta=[0.75, 0.95],
                                   verbose=False)
        else:
            raise ValueError(f'Unknown source: {source}')

        elapsed = _time.time() - t0
        if verbose:
            print(f'Done. {pool_size} samples in {elapsed:.1f}s '
                  f'({pool_size/elapsed:.0f} samples/s)')

        # 存入内存
        if reshape_for_resnet:
            N = self.T_obs * self.fs
            start = N // 4
            end = start + self.fs * self.T
            self.pool_strains = ts[0][:, :, start:end].astype(np.float32)
        else:
            self.pool_strains = np.expand_dims(ts[0], 1).astype(np.float32)

        self.pool_labels = ts[1]
        self.pool_size = len(self.pool_labels)

        if verbose:
            mem_mb = self.pool_strains.nbytes / 1024**2
            print(f'Pool in memory: {self.pool_strains.shape}, {mem_mb:.0f} MB')

        # 初始抽一批
        self._resample()

    def _resample(self):
        """从池子里随机抽 nsample_perepoch 个 (有放回)"""
        idx = np.random.choice(self.pool_size, size=self.nsample_perepoch, replace=True)
        self.strains = self.pool_strains[idx]
        self.labels = self.pool_labels[idx]

    def generate(self, nsample=None, verbose=False):
        """兼容旧接口: 每 epoch 调一次, 实际只做 resample"""
        if nsample is not None:
            self.nsample_perepoch = nsample
        self._resample()

    def __len__(self):
        return len(self.strains)

    def __getitem__(self, idx):
        return self.strains[idx], self.labels[idx]


# ============================================================
# 模型加载/保存
# ============================================================
def load_model(checkpoint_dir, model_class, prefer='last'):
    """加载模型, 优先加载最终 epoch, 兼容旧版 best checkpoint"""
    net = model_class()
    p = Path(checkpoint_dir)
    loss_file = p / 'train_loss_history.npy'
    train_loss_history = np.load(loss_file).tolist() if loss_file.exists() else []

    if p.is_dir():
        if prefer == 'best':
            patterns = ['best_model*.pt', 'last_model*.pt', '*.pt']
        else:
            patterns = ['last_model*.pt', 'best_model*.pt', '*.pt']

        for pattern in patterns:
            files = sorted(p.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
            if files:
                checkpoint_path = files[0]
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                net.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 0)
                print(f'Loaded model from {checkpoint_path}, epoch={epoch}')
                return net, epoch, train_loss_history

    print('Init new network!')
    return net, 0, train_loss_history


def save_model(epoch, model, optimizer, scheduler, checkpoint_dir, train_loss_history, filename, delete_pattern=None):
    p = Path(checkpoint_dir)
    p.mkdir(parents=True, exist_ok=True)
    if delete_pattern is not None:
        for f in p.glob(delete_pattern):
            if f.name != filename:
                os.remove(f)
    np.save(p / 'train_loss_history', train_loss_history)
    output = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    if scheduler is not None:
        output['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(output, p / filename)


# ============================================================
# 训练循环
# ============================================================
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    return float((y_hat.type(y.dtype) == y).sum())


def train(net, dataset_train, dataset_test, args, checkpoint_dir, device):
    """训练主循环"""
    data_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loss_history = []
    best_test_loss = float('inf')

    for epoch in range(args.epochs):
        # 每 epoch 重新生成数据
        dataset_train.generate(args.nsample, verbose=False)

        net.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for x, y in data_loader:
            x = x.to(device).float()
            y = y.to(device).long()

            optimizer.zero_grad()
            pred = net(x)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += loss.item() * x.size(0)
                total_correct += accuracy(pred, y)
                total_samples += x.size(0)

        scheduler.step()

        # 评估
        net.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device).float()
                y = y.to(device).long()
                pred = net(x)
                loss = loss_func(pred, y)
                test_loss += loss.item() * x.size(0)
                test_correct += accuracy(pred, y)
                test_total += x.size(0)

        train_l = total_loss / total_samples
        train_acc = total_correct / total_samples
        test_l = test_loss / test_total
        test_acc = test_correct / test_total

        train_loss_history.append([epoch + 1, train_l, test_l, train_acc, test_acc])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:3d}/{args.epochs} | '
                  f'train_loss={train_l:.4f} test_loss={test_l:.4f} | '
                  f'train_acc={train_acc:.3f} test_acc={test_acc:.3f} | '
                  f'lr={scheduler.get_last_lr()[0]:.6f}')

        if test_l < best_test_loss:
            best_test_loss = test_l
            save_model(epoch + 1, net, optimizer, scheduler,
                       checkpoint_dir, train_loss_history,
                       f'best_model_e{epoch + 1}.pt',
                       delete_pattern='best_model*.pt')

    save_model(args.epochs, net, optimizer, scheduler,
               checkpoint_dir, train_loss_history,
               f'last_model_e{args.epochs}.pt',
               delete_pattern='last_model*.pt')

    print(f'\nTraining done. Best test loss = {best_test_loss:.4f}')
    return train_loss_history


# ============================================================
# 评估 + ROC
# ============================================================
def evaluate_gpu(net, data_iter, device):
    """在 GPU 上计算预测概率和真实标签"""
    net.eval()
    softmax = nn.Softmax(dim=-1)
    y_hat_list, y_list = [], []

    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device).float()
            y = y.to(device).long()
            y_hat = net(X)
            preds = softmax(y_hat).cpu().numpy()[:, 1].tolist()
            labels = y.cpu().numpy().tolist()
            y_hat_list.extend(preds)
            y_list.extend(labels)

    return np.array(y_hat_list), np.array(y_list)


def plot_roc(net, source, reshape_for_resnet, device, outdir, fs=8192, T=1):
    """画不同 SNR 下的 ROC 曲线"""
    colors = cycle(["deeppink", "aqua", "darkorange", "cornflowerblue"])
    snr_list = [5, 10, 15, 20]

    plt.figure(figsize=(8, 6))

    for snr in tqdm(snr_list, desc='Evaluating SNRs'):
        dataset = GWDataset(source=source, fs=fs, T=T, snr=snr,
                            pool_size=1000, nsample_perepoch=1000,
                            Nnoise=25,
                            reshape_for_resnet=reshape_for_resnet,
                            verbose=False)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        y_hat, y_true = evaluate_gpu(net, loader, device)

        fpr, tpr, _ = roc_curve(y_true, y_hat)
        auc_val = roc_auc_score(y_true, y_hat)
        plt.plot(fpr, tpr, color=next(colors), label=f'SNR={snr} (AUC={auc_val:.2f})')

    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--', label='Luck (AUC=0.50)')
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curves — {source.upper()}')
    plt.legend()
    plt.tight_layout()

    outpath = Path(outdir) / f'roc_{source}.png'
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f'ROC plot saved to {outpath}')


def plot_loss_history(history, outdir, label=''):
    """画 loss 曲线"""
    history = np.array(history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history[:, 0], history[:, 1], label='train loss')
    ax1.plot(history[:, 0], history[:, 2], label='test loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title(f'Loss — {label}')

    ax2.plot(history[:, 0], history[:, 3], label='train acc')
    ax2.plot(history[:, 0], history[:, 4], label='test acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title(f'Accuracy — {label}')

    plt.tight_layout()
    outpath = Path(outdir) / f'loss_history_{label}.png'
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f'Loss plot saved to {outpath}')


# ============================================================
# Main
# ============================================================
def main():
    args = get_args()

    # device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')

    # 输出目录
    if args.outdir is None:
        args.outdir = f'./checkpoints_{args.phase}/'
    os.makedirs(args.outdir, exist_ok=True)

    # ========== Phase 1: BBH baseline ==========
    if args.phase == 'baseline':
        print('=' * 60)
        print('Phase 1: BBH baseline (原版 CNN)')
        print('=' * 60)

        # 导入原版模型
        sys.path.insert(0, '.')
        try:
            from main import MyNet
        except ModuleNotFoundError:
            baseline_dir = Path(__file__).resolve().parent / 'GWData-Bootcamp-main' / '2023' / 'deep_learning' / 'baseline'
            sys.path.insert(0, str(baseline_dir))
            from main import MyNet

        source = 'bbh'
        reshape = False

        dataset_train = GWDataset(source=source, snr=args.snr_train,
                                  pool_size=args.pool_size,
                                  nsample_perepoch=args.nsample,
                                  reshape_for_resnet=reshape)
        dataset_test = GWDataset(source=source, snr=args.snr_train,
                                 pool_size=args.pool_size // 5,
                                 nsample_perepoch=args.nsample,
                                 reshape_for_resnet=reshape)

        net, start_epoch, history = load_model(args.outdir, MyNet)
        net.to(device)
        print(f'MyNet params: {sum(p.numel() for p in net.parameters()):,}')

        history = train(net, dataset_train, dataset_test, args, args.outdir, device)
        plot_loss_history(history, args.outdir, label='bbh_cnn')
        plot_roc(net, source, reshape, device, args.outdir)

    # ========== Phase 2: BNS + ResNet ==========
    elif args.phase == 'bns_resnet':
        print('=' * 60)
        print('Phase 2: BNS + ResNet')
        print('=' * 60)

        from model_resnet_v2 import ResNet1D, ResNet1D_Small

        source = 'bns'
        reshape = True  # 用 (ndet, N) 格式给 1D ResNet

        if args.model == 'full':
            model_class = ResNet1D
        else:
            model_class = ResNet1D_Small

        dataset_train = GWDataset(source=source, snr=args.snr_train,
                                  pool_size=args.pool_size,
                                  nsample_perepoch=args.nsample,
                                  reshape_for_resnet=reshape)
        dataset_test = GWDataset(source=source, snr=args.snr_train,
                                 pool_size=args.pool_size // 5,
                                 nsample_perepoch=args.nsample,
                                 reshape_for_resnet=reshape)

        net, start_epoch, history = load_model(args.outdir, model_class)
        net.to(device)
        print(f'{model_class.__name__} params: {sum(p.numel() for p in net.parameters()):,}')

        history = train(net, dataset_train, dataset_test, args, args.outdir, device)
        plot_loss_history(history, args.outdir, label='bns_resnet')
        plot_roc(net, source, reshape, device, args.outdir)


if __name__ == '__main__':
    main()
