"""Per-modality Gaussian experts and their variants (plain, ensemble, MC-dropout, spectral-norm,
corruption-augmented). Each expert maps a clip's pooled statistics to (mu, var) of the label."""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)


def pool(x):
    """Pooled clip statistics: mean and std over time (what most MOSEI pipelines feed a head)."""
    return np.concatenate([x.mean(1), x.std(1)], axis=1).astype(np.float32)


class Net(nn.Module):
    def __init__(self, d, h=64, p=0.0, sn=False):
        super().__init__()
        lin = (lambda a, b: nn.utils.parametrizations.spectral_norm(nn.Linear(a, b))) if sn else nn.Linear
        self.body = nn.Sequential(lin(d, h), nn.GELU(), nn.Dropout(p), lin(h, h), nn.GELU(), nn.Dropout(p))
        self.out = nn.Linear(h, 2)

    def forward(self, x):
        z = self.body(x)
        o = self.out(z)
        return o[:, 0], o[:, 1].clamp(-7, 5), z


def _nll(mu, lv, y):
    return 0.5 * (lv + (y - mu) ** 2 / lv.exp()).mean()


class Expert:
    """kind: plain | ens | mcd | sn | aug (corruption-augmented, self-supervised corruptions only)."""

    def __init__(self, kind='plain', n_ens=5, seed=0, epochs=400):
        self.kind, self.n_ens, self.seed, self.epochs = kind, n_ens, seed, epochs

    def fit(self, x_train, y, x_val, y_val, swap_pool=None):
        f, fv = pool(x_train), pool(x_val)
        self.m, self.s = f.mean(0), f.std(0) + 1e-6
        self.xmu, self.xsd = x_train.mean((0, 1)), x_train.std((0, 1)) + 1e-6
        n_models = self.n_ens if self.kind == 'ens' else 1
        self.nets = []
        for k in range(n_models):
            torch.manual_seed(self.seed * 100 + k)
            net = Net(f.shape[1], p=0.2 if self.kind == 'mcd' else 0.0, sn=self.kind == 'sn')
            self.nets.append(self._train(net, x_train, y, fv, y_val))
        return self

    def _feats(self, x):
        return torch.from_numpy((pool(x) - self.m) / self.s)

    def _train(self, net, x_train, y, fv, y_val):
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        Xt = self._feats(x_train)
        Yt, Xv, Yv = torch.from_numpy(y), torch.from_numpy((fv - self.m) / self.s), torch.from_numpy(y_val)
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = 1e9, None, 0
        for ep in range(self.epochs):
            net.train()
            X = Xt
            if self.kind == 'aug':                      # [self-sup] white noise or swap on 30% of clips
                xb = x_train.copy()
                idx = rng.random(len(xb)) < 0.3
                half = rng.random(len(xb)) < 0.5
                w = idx & half
                xb[w] = (self.xmu + self.xsd * rng.normal(size=xb[w].shape)).astype(np.float32)
                sw = np.where(idx & ~half)[0]
                xb[sw] = x_train[rng.permutation(len(x_train))[:len(sw)]]
                X = self._feats(xb)
            mu, lv, _ = net(X)
            loss = _nll(mu, lv, Yt)
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 10 == 0:
                net.eval()
                with torch.no_grad():
                    mv, lvv, _ = net(Xv)
                    v = _nll(mv, lvv, Yv).item()
                if v < best - 1e-4:
                    best, bad = v, 0
                    best_state = {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                    if bad > 6:
                        break
        net.load_state_dict(best_state)
        net.eval()
        return net

    def predict(self, x, n_mc=20):
        """Returns mu, var, hidden features (first member)."""
        X = self._feats(x)
        with torch.no_grad():
            if self.kind == 'mcd':
                net = self.nets[0]
                net.train()
                outs = [net(X) for _ in range(n_mc)]
                net.eval()
                mus = torch.stack([o[0] for o in outs]); vs = torch.stack([o[1].exp() for o in outs])
                _, _, z = net(X)
            else:
                outs = [n(X) for n in self.nets]
                mus = torch.stack([o[0] for o in outs]); vs = torch.stack([o[1].exp() for o in outs])
                z = outs[0][2]
        mu = mus.mean(0)
        var = vs.mean(0) + mus.var(0, unbiased=False)
        return mu.numpy().astype(np.float64), var.numpy().astype(np.float64), z.numpy()
