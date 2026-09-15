"""Small reproducible regression experiment. Run from the project directory."""
import argparse
import copy
import csv
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from torch import nn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_utils import load_data, split_data, FEATURES

def scores(y, prediction):
    return {'rmse_db': float(np.sqrt(mean_squared_error(y, prediction))),
            'mae_db': float(mean_absolute_error(y, prediction)),
            'r2': float(r2_score(y, prediction))}

def make_model():
    return nn.Sequential(nn.Linear(5, 32), nn.ReLU(),
                         nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

def fit_network(Xtr, ytr, Xva, yva, seed, epochs, weight_decay, target_scale):
    torch.manual_seed(seed)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    tx, ty = torch.tensor(Xtr), torch.tensor(ytr[:, None])
    vx, vy = torch.tensor(Xva), torch.tensor(yva[:, None])
    best_loss, best_state, best_epoch = float('inf'), None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in torch.randperm(len(tx)).split(64):
            optimizer.zero_grad()
            loss = loss_fn(model(tx[batch]), ty[batch])
            loss.backward()
            optimizer.step()
        # Compare both partitions in evaluation mode at the same epoch.
        model.eval()
        with torch.no_grad():
            training = loss_fn(model(tx), ty).item()
            validation = loss_fn(model(vx), vy).item()
        if not np.isfinite(training + validation):
            raise RuntimeError('Non-finite loss; check input and learning rate')
        history.append([epoch, np.sqrt(training) * target_scale,
                        np.sqrt(validation) * target_scale])
        if validation < best_loss:
            best_loss = validation
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    # Continue to fixed budget for diagnostic curves, then restore best validation epoch.
    # This is checkpoint selection, not an early-termination training loop.
    model.load_state_dict(best_state)
    return model, np.array(history), best_epoch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/airfoil_self_noise.dat')
    parser.add_argument('--out', default='outputs')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 17, 27])
    parser.add_argument('--evaluate-test', action='store_true',
                        help='Use only after freezing experiments and report plan')
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Epochs must be positive and seeds unique')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, y, checksum = load_data(args.data)
    tr, va, te, groups = split_data(X)
    sx, sy = StandardScaler().fit(X[tr]), StandardScaler().fit(y[tr, None])
    Z = sx.transform(X).astype('float32')
    target = sy.transform(y[:, None]).ravel().astype('float32')
    scale = float(sy.scale_[0])
    rows, models, predictions, validation_rmse = [], {}, {}, {}
    partitions = {'train': tr, 'validation': va}
    if args.evaluate_test:
        partitions['test'] = te

    def record(name, seed, predict, best_epoch=None):
        for split, ids in partitions.items():
            pred = predict(Z[ids])
            result = scores(y[ids], pred)
            rows.append(dict(model=name, seed=seed, split=split,
                             best_epoch=best_epoch, **result))
            if split == 'validation':
                validation_rmse.setdefault(name, []).append(result['rmse_db'])
            if split == 'test':
                predictions[f'{name}_{seed}'] = pred

    for name, estimator in [('mean', DummyRegressor()), ('linear', LinearRegression())]:
        estimator.fit(Z[tr], y[tr])
        record(name, '', estimator.predict)

    # Same architecture, initial seed, batch order and budget; vary only weight decay.
    for seed in args.seeds:
        for name, decay in [('mlp_plain', 0.), ('mlp_l2', .001)]:
            model, history, epoch = fit_network(Z[tr], target[tr], Z[va], target[va],
                                                seed, args.epochs, decay, scale)
            def predict(z):
                model.eval()
                with torch.no_grad():
                    p = model(torch.tensor(z)).numpy()
                return sy.inverse_transform(p).ravel()
            record(name, seed, predict, epoch)
            np.savetxt(out / f'{name}_{seed}_history.csv', history, delimiter=',',
                       header='epoch,train_rmse_db,validation_rmse_db', comments='')
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(history[:, 0], history[:, 1], label='Train')
            ax.plot(history[:, 0], history[:, 2], label='Validation')
            ax.axvline(epoch, linestyle=':', color='grey', label='Selected epoch')
            ax.set(xlabel='Epoch', ylabel='RMSE (dB)', title=f'{name}, seed {seed}')
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f'{name}_{seed}_learning.png', dpi=160)
            plt.close(fig)
            torch.save(model.state_dict(), out / f'{name}_{seed}.pt')

    # Selection uses validation only, regardless of whether test is exposed.
    chosen = min(validation_rmse, key=lambda n: np.mean(validation_rmse[n]))
    with (out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for name in validation_rmse:
        summary[name] = {}
        for split in partitions:
            entries = [r for r in rows if r['model'] == name and r['split'] == split]
            summary[name][split] = {
                metric: {'mean': float(np.mean([r[metric] for r in entries])),
                         'sd': float(np.std([r[metric] for r in entries], ddof=1))
                         if len(entries) > 1 else None}
                for metric in ['rmse_db', 'mae_db', 'r2']}
    metadata = dict(arguments=vars(args), data_sha256=checksum,
                    selected_model_family=chosen,
                    selection='Lowest mean validation RMSE across initialisation seeds',
                    split_seed=42, validation_split_seed=43,
                    rows={s: len(i) for s, i in [('train', tr), ('validation', va), ('test', te)]},
                    groups={s: len(set(groups[i])) for s, i in [('train', tr), ('validation', va), ('test', te)]},
                    versions={'python': platform.python_version(), 'torch': str(torch.__version__),
                              'numpy': np.__version__, 'sklearn': sklearn.__version__},
                    architecture='5-32-32-1, ReLU', features=FEATURES)
    (out / 'run.json').write_text(json.dumps(metadata, indent=2))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    np.savez(out / 'preprocessing_and_split.npz', x_mean=sx.mean_, x_scale=sx.scale_,
             y_mean=sy.mean_, y_scale=sy.scale_, train=tr, validation=va, test=te)
    if args.evaluate_test:
        with (out / 'test_predictions.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source_row_zero_based', 'actual_db', *predictions])
            writer.writerows(zip(te, y[te], *predictions.values()))
        # Preselected first seed for a plot; never choose the best test seed.
        key = f'{chosen}_{args.seeds[0]}' if chosen.startswith('mlp') else f'{chosen}_'
        pred = predictions[key]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(y[te], pred, s=12, alpha=.5)
        lo, hi = min(y[te].min(), pred.min()), max(y[te].max(), pred.max())
        axes[0].plot([lo, hi], [lo, hi], 'k--')
        axes[0].set(xlabel='Measured (dB)', ylabel='Predicted (dB)')
        axes[1].scatter(y[te], pred - y[te], s=12, alpha=.5)
        axes[1].axhline(0, color='black', linestyle='--')
        axes[1].set(xlabel='Measured (dB)', ylabel='Prediction − measurement (dB)')
        fig.suptitle(f'Validation-selected model: {key}')
        fig.tight_layout()
        fig.savefig(out / 'test_diagnostics.png', dpi=160)
        plt.close(fig)
    print('Validation-selected model family:', chosen)
    for name, item in summary.items():
        split = 'test' if args.evaluate_test else 'validation'
        print(name, split, item[split]['rmse_db'])
    print('Saved:', out.resolve())

if __name__ == '__main__':
    main()
