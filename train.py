from pareto.model import ResNet
import tinygrad.nn as nn
from tinygrad import dtypes, Tensor, Context, GlobalCounters, TinyJit
from tinygrad.helpers import trange, getenv, BEAM
import wandb
import time
import numpy as np
from tinygrad.device import Device
from pareto.extra import OneCycleLR
from functools import partial
import uuid
import json
import pathlib

def augment_flip(batch: Tensor) -> Tensor:
    return Tensor.where(Tensor.rand_like(batch[:, 0, 0, 0].view(-1, 1, 1, 1)) < 0.5, Tensor.flip(batch, (-1,)), batch)

def augment_pad_crop(batch: Tensor) -> Tensor:
    r: int = 4
    n, _, h, w = batch.shape
    batch_padded = batch.pad((r, r, r, r), mode="reflect")
    shifts = Tensor.randint((len(batch), 2), low=-r, high=r + 1)
    rows = batch_padded[:, :, 0:h, :]
    for i in range(1, 2 * r + 1):
        rows = (shifts[:, 0] == i - 4).reshape(n, 1, 1, 1).where(batch_padded[:, :, i : i + h, :], rows)
    rows = rows.contiguous()
    out = rows[:, :, :, 0:w]
    for i in range(1, 2 * r + 1):
        out = (shifts[:, 1] == i - 4).reshape(n, 1, 1, 1).where(rows[:, :, :, i : i + w], out)
    return out

def augment_cutout(batch: Tensor) -> Tensor:
    cut = 8
    n, _, h, w = batch.shape
    rows = Tensor.randint(n, low=0, high=h - cut).reshape(n, 1, 1, 1)
    cols = Tensor.randint(n, low=0, high=w - cut).reshape(n, 1, 1, 1)
    rr, cc = Tensor.arange(h).reshape(1, 1, h, 1), Tensor.arange(w).reshape(1, 1, 1, w)
    hole = (rr >= rows) & (rr < rows + cut) & (cc >= cols) & (cc < cols + cut)
    return (hole).where(0, batch)

def augment(batch: Tensor):
    return augment_cutout(augment_flip(augment_pad_crop(batch)))

def main(cs:float, warmup:bool = False):
    X_train, Y_train, X_test, Y_test = nn.datasets.cifar()
    X_train, Y_train, X_test, Y_test = X_train.numpy(), Y_train.numpy(), X_test.numpy(), Y_test.numpy()

    cifar10_mean, cifar10_std = Tensor([0.4914, 0.4822, 0.4465]), Tensor([0.2470, 0.2435, 0.2616])

    def preprocess(X: Tensor) -> Tensor:
        return ((X / 255.0 - cifar10_mean.view(1, -1, 1, 1)) / cifar10_std.view(1, -1, 1, 1)).cast(dtypes.default_float)

    def batch(bs):
        samples = np.random.randint(low=0, high=X_train.shape[0], size=bs)
        return (Tensor(X_train[samples], device="NPY"), Tensor(Y_train[samples], device="NPY"))

    @Context(TRAINING=1)
    def train_step(model, opt, scheds, scaler, X, Y, fq) -> Tensor:
        opt.zero_grad()
        X, Y = X.to(Device.DEFAULT), Y.to(Device.DEFAULT)
        with Context(FAKEQUANT=1 if fq else 0):
            loss = model(augment(preprocess(X))).cast(dtypes.float32).sparse_categorical_crossentropy(Y, label_smoothing=0.2)
            (loss*scaler).backward()
        for t in opt.params: t.grad = t.grad.contiguous() / scaler
        return loss.realize(*opt.schedule_step(), *scheds[0].schedule_step(), *scheds[1].schedule_step(), *model.ema_tensors())

    @Context(TRAINING=0)
    def eval_step(model, X, Y, eval_bs, fq) -> tuple[Tensor, Tensor]:
        losses, accs = [], []
        for batch in range(0, len(X), eval_bs):
            Xv, Yv = (
                Tensor(X[batch : batch + eval_bs], device="NPY").to(Device.DEFAULT),
                Tensor(Y[batch : batch + eval_bs], device="NPY").to(Device.DEFAULT),
            )
            with Context(FAKEQUANT=1 if fq else 0):
                logits = model(preprocess(Xv)).cast(dtypes.float32)
            losses.append(logits.sparse_categorical_crossentropy(Yv, label_smoothing=0.2))
            accs.append((logits.softmax(axis=1).argmax(axis=1) == Yv).mean())
        return Tensor.stack(*losses).mean(), Tensor.stack(*accs).mean()

    quantized=bool(getenv('QUANTIZED', 0))
    qflags = (0,1) if quantized else (0,)
    train_steps = [TinyJit(partial(train_step, fq=fq)) for fq in qflags]
    eval_steps = [TinyJit(partial(eval_step, fq=fq)) for fq in qflags]

    model = ResNet(cs=cs,quantized=quantized)
    bs = 512
    cc,mc = model.compute_costs(bs)
    wd, nowd = [], []
    for param in nn.state.get_parameters(model):
        if param.ndim == 1:
            nowd.append(param)
        else:
            wd.append(param)

    opt_wd = nn.optim.SGD(wd, momentum=0.9, nesterov=True, weight_decay=5e-4)
    opt_nowd = nn.optim.SGD(nowd, momentum=0.9, nesterov=True, weight_decay=0.0)

    epochs = 20
    steps_per_epoch = len(X_train) // bs
    total_train_steps = epochs * steps_per_epoch
    initial_div_factor = 1e16
    bias_lr_scale = 64
    max_lr = 0.6
    final_lr_ratio = 0.001
    pct_start = 0.20
    quant_start = int(total_train_steps * 0.2)

    sched = partial(
        OneCycleLR,
        pct_start=pct_start,
        div_factor=initial_div_factor,
        final_div_factor=1.0 / (initial_div_factor * final_lr_ratio),
        total_steps=total_train_steps,
    )
    sched_nowd, sched_wd = sched(opt_nowd, max_lr=max_lr * bias_lr_scale), sched(opt_wd, max_lr=max_lr)
    loss_scaler = 32.0
    if warmup:
        for _ in trange(5):
            x, y = batch(bs)
            st = time.perf_counter()
            for idx in qflags:
                loss = train_steps[idx](model, nn.optim.OptimizerGroup(opt_wd, opt_nowd), [sched_nowd, sched_wd], loss_scaler, x, y)
            Device[Device.DEFAULT].synchronize()
            et = time.perf_counter()

        for _ in trange(5):
            for idx in qflags:
                val_loss, val_acc = eval_steps[idx](model, X_test, Y_test, bs)
        return

    wandb.init(
        project="tiny-amd",
        config={
            "flags": {"DEV": Device.default.__repr__(), "JITBEAM": getenv("JITBEAM", BEAM.value), "WINO": getenv("WINO", 0)},
            "model": {"channels":model.channels,"quantized":quantized, "c": cs, 'compute_cost': cc, 'memory_cost': mc},
        },
    )
    fwd_bwd_flops = model.compute_fwd_bwd_flops(bs) / 1e12
    for i in trange(total_train_steps):
        fq = int((i > quant_start) and quantized)
        GlobalCounters.reset()
        x, y = batch(bs)
        st = time.perf_counter()
        loss = train_steps[fq](model, nn.optim.OptimizerGroup(opt_wd, opt_nowd), [sched_nowd, sched_wd], loss_scaler, x, y)
        Device[Device.DEFAULT].synchronize()
        et = time.perf_counter()
        step_time = et - st
        stats = {
            "train/loss": loss.item(),
            "train/step_time": step_time,
            "train/lr": opt_wd.lr.numpy(),
            "train/tflops": fwd_bwd_flops / step_time,
        }

        if (i % steps_per_epoch == 0):
            val_loss, val_acc = eval_steps[fq](model, X_test, Y_test, bs)
            stats |= {"val/acc": val_acc.item() * 100, "val/loss": val_loss.item()}
        wandb.log(stats)

    val_loss, val_acc = eval_steps[fq](model, X_test, Y_test, bs)
    stats = {"val/acc": val_acc.item() * 100, "val/loss": val_loss.item()}
    wandb.log(stats)
    wandb.finish()
    perf={'compute_cost': cc, 'mem_cost': mc, 'accuracy':val_acc.item() * 100,'c':cs,'quantized':quantized}
    savepath=pathlib.Path(f'results/{uuid.uuid4()}.json')
    savepath.parent.mkdir(parents=True,exist_ok=True)
    with savepath.open(mode='w') as f:
        json.dump(perf,f,indent=2)

if __name__ == "__main__":
    cs,trials = getenv('SCALE', 1.0),getenv('TRIALS', 10)
    main(cs, warmup=True)
    [main(cs) for _ in range(trials)]