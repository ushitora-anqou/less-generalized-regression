from chainer.dataset import convert
from chainer.training import extension
from chainer import variable
from chainer import cuda
from chainer import training, serializers
import os
import copy
import six
import matplotlib.pyplot as plt
import tempfile
import shutil
from collections import deque



class HyperEvaluator(extension.Extension):
    trigger = 1, 'epoch'

    def __init__(self, iterators, converter=convert.concat_examples, device=None):
        self._iterators = iterators

        self.converter = converter
        self.device = device

    def __call__(self, trainer):
        data = []
        target = trainer.updater.get_optimizer('main').target
        for label, iterator in self._iterators:
            it = copy.copy(iterator)
            if hasattr(iterator, 'reset'):
                it.reset()
            it._repeat = False

            for batch in it:
                in_arrays = self.converter(batch, self.device)

                if not isinstance(in_arrays, tuple):
                    raise RuntimeError('invalid argument: not supported type')

                src = variable.Variable(in_arrays[0], volatile='on')
                res = target.predictor(src)
                res_data = cuda.to_cpu(res.data)

                for i, s in enumerate(batch):
                    x, t = s[0:2]
                    y = res_data[i]
                    data.append({
                        'x': x,
                        'y': y,
                        't': t,
                        'label': label})

        self.process(trainer, data)

    def initialize(self, trainer):
        pass

    def process(self, trainer, data):
        raise NotImplementedError()


class PlotFunc(HyperEvaluator):
    def __init__(self, train_iter, test_iter, filename, gpu, range_x, range_y):
        super().__init__(
            iterators=[('train', train_iter), ('test', test_iter)],
            device=gpu
        )
        self.filename = filename
        # target area to plot
        self.xlim_min = range_x[0]
        self.xlim_max = range_x[1]
        self.ylim_min = range_y[0]
        self.ylim_max = range_y[1]

    def process(self, trainer, data):
        plt.cla()
        plt.grid(which='major', color='gray', linestyle='-')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(self.xlim_min, self.xlim_max)
        plt.ylim(self.ylim_min, self.ylim_max)

        src = sorted(data, key=lambda d: d['x'][0])
        train_x = [d['x'] for d in src if d['label'] == 'train']
        test_x = [d['x'] for d in src if d['label'] == 'test']
        train_y = [d['y'] for d in src if d['label'] == 'train']
        test_y = [d['y'] for d in src if d['label'] == 'test']
        train_t = [d['t'] for d in src if d['label'] == 'train']
        test_t = [d['t'] for d in src if d['label'] == 'test']

        plt.plot(test_x, test_t, label='ans')
        plt.plot(train_x, train_y, label='train')
        plt.plot(test_x, test_y, label='test')
        plt.plot(train_x, train_t, '.', label='sample value')
        # 1-origin
        current_epoch = trainer.updater.epoch
        plt.title('epoch = {}'.format(current_epoch))
        plt.legend()
        path = os.path.join(trainer.out, self.filename.format(trainer))
        plt.savefig(path)



class LatestSnapshot(training.Extension):
    def __init__(self,
                 target,
                 savefun=serializers.save_npz,
                 filename='snapshot_iter_{.updater.iteration}',
                 savefiles=1):
        self.savefun = savefun
        self.filename = filename
        self.savefiles = savefiles
        # 保存したスナップショット一覧
        # Resume したとき、Resume 前の部分が消えないけど、これはもはや仕様か
        self.saved_snapshots = deque([])
        # 参照のみ取ってくる
        self.target = target

    def __call__(self, trainer):
        ### Retrieved from Chainer source code: 
        ### https://docs.chainer.org/en/v1.24.0/_modules/chainer/training/extensions/_snapshot.html
        ### License: MIT
            #Copyright (c) 2015 Preferred Infrastructure, Inc.
            #Copyright (c) 2015 Preferred Networks, Inc.
            #Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

            #    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

            #    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
        ### from here
        fname_complete = self.filename.format(trainer)
        prefix = 'tmp' + fname_complete
        fd, temp_filename = \
            tempfile.mkstemp(prefix=prefix, dir=trainer.out, text=False)
        try:
            self.savefun(temp_filename, self.target)
        except Exception:
            os.close(fd)
            os.remove(temp_filename)
            raise

        os.close(fd)
        fpath_complete = os.path.join(trainer.out, fname_complete)
        shutil.move(temp_filename, fpath_complete)
        ### end of Chainer's code

        self.saved_snapshots.append(fpath_complete)
        if (self.savefiles > 0) and len(self.saved_snapshots) > self.savefiles:
            ObsoleteFile = self.saved_snapshots.popleft()
            try:
                # KeyBoard Interrupt があってもファイルは削除されるので問題ない
                os.remove(ObsoleteFile)
            except OSError:
                print("Notice: Snapshot", ObsoleteFile, "was already removed.")
                # ignore
