import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, training, report
from chainer import Link, Chain, ChainList
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import time
from matplotlib import pyplot as plt
import os
from more_extensions import PlotFunc, LatestSnapshot


class AttackUpdater(training.StandardUpdater):
    def cleargrads(self):
        self._optimizers['main'].target.cleargrads()

    # override
    def update_core(self):
        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        train_batch = self._iterators['main'].next()
        train_args = [Variable(x) for x in self.converter(train_batch, self.device)]

        attack_batch = self.attack_iter.next()
        attack_args = [Variable(x) for x in self.converter(attack_batch, self.device)]

        self.cleargrads()
        (loss_func(*attack_args, False) * self.gamma).backward()
        loss_func(*train_args).backward()
        optimizer.update()


class Regression(Chain):
    def __init__(self, predictor):
        super(Regression, self).__init__(predictor=predictor)

    def __call__(self, x, t, enable_report=True):
        y = self.predictor(x)
        loss = F.mean_squared_error(y, t)
        if enable_report:
            report({'loss': loss}, self)
        return loss


class MLP(Chain):
    def __init__(self):
        initW = chainer.initializers.HeNormal()
        super(MLP, self).__init__(
            l1=L.Linear(1, 512, initialW=initW),
            l2=L.Linear(512, 512, initialW=initW),
            l3=L.Linear(512, 1, initialW=initW)
        )

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h


def get_dataset(N):
    x = np.linspace(-np.pi, +np.pi, N, dtype=np.float32).reshape(-1, 1)
    t = np.sin(x)
    return x, t


if __name__ == '__main__':
    # train samples
    N = 16
    train_batch_size = 4
    x_train, t_train = get_dataset(N + 1)
    train_iter = chainer.iterators.SerialIterator(chainer.datasets.TupleDataset(x_train, t_train), train_batch_size)

    # test samples
    N_test = 500
    test_batch_size = 500
    x_test, t_test = get_dataset(N_test)
    test_iter = chainer.iterators.SerialIterator(chainer.datasets.TupleDataset(x_test, t_test), test_batch_size, False, False)

    # attack samples
    N_attack = 16
    attack_batch_size = 4
    x_attack = x_train + np.pi / 32
    t_attack = np.sin(x_attack)
    #x_attack, t_attack = get_dataset(N_attack)
    np.random.shuffle(t_attack)
    attack_iter = chainer.iterators.SerialIterator(chainer.datasets.TupleDataset(x_attack, t_attack), attack_batch_size)

    # parameters
    gamma = 0.1
    epoch = 1502
    out = 'output'
    gpu_id = -1
    resume_filepath = None

    # make trainer
    model = Regression(MLP())
    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = AttackUpdater(train_iter, optimizer, device=gpu_id)
    updater.gamma = gamma
    updater.attack_iter = attack_iter
    #updater = chainer.training.StandardUpdater(train_iter, optimizer, device=gpu_id)

    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png', marker=None))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch:06d}.npz'), trigger=(10, 'epoch'))
    trainer.extend(PlotFunc(train_iter, test_iter, 'plot-{.updater.epoch:06d}.png', gpu_id, [-np.pi, np.pi], [-1, 1]), trigger=(10, 'epoch'))
    trainer.extend(LatestSnapshot(trainer))

    if resume_filepath:
        serializers.load_npz(resume_filepath, trainer)

    trainer.run()

