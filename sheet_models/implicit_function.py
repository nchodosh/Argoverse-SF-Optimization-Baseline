import inspect
from pathlib import Path

import numpy as np
import torch
import utils.live_log
import utils.options
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm, trange
from utils.geometry import yp_to_unit

from . import base

#from utils.live_log import log

log = None

EVAL_CHUNK_SIZE = 20000
MAX_DEPTH = 80

class WorldSheet(base.WorldSheet):
    def __init__(self, opt):
        super().__init__(opt)
        self.parameters_suffix = '.pt'
        self.opt = opt
        self.graph = Graph(opt)
        self.graph.to(opt.device)
        self.scheduler = getattr(torch.optim.lr_scheduler, opt.optim.sched.type)


    def fit(self, pcl_train, pcl_test=None, quiet=False):
        pcl_train = pcl_train.float().to(self.opt.device)
        if pcl_test is not None:
            pcl_test = pcl_test.float().to(self.opt.device)
            
        self.graph.define_network()
        self.graph.to(self.opt.device)
        optimizer = getattr(torch.optim, self.opt.optim.type)(self.graph.parameters(), lr=self.opt.optim.lr)

        sched_args = utils.options.namespace_to_dict(self.opt.optim.sched.args)
        available_sched_args = inspect.getfullargspec(self.scheduler).args
        to_remove = [key for key in sched_args if key not in available_sched_args]
        for key in to_remove:
            sched_args.pop(key)
        
        schedule = self.scheduler(optimizer,
                                  **sched_args)

        if self.opt.out.invert_depth:
            loss_fn = lambda p, t: (p - 1.0/t).abs()
            to_depth = lambda x: 1/x
        else:
            loss_fn = lambda p, t: (p - t).abs()
            to_depth = lambda x: x
        pred = torch.ones(1)
        test_err = torch.zeros(1)
        close = (pcl_train[:, 0].abs() < 50)

        if not quiet:
            utils.live_log.init()
            log = utils.live_log.log
            prog = Progress(TextColumn('[green]{task.completed}/{task.total} [red]{task.fields[loss]:.3g}, [blue]LR: {task.fields[lr]:.2g}, Md: {task.fields[median]:.2g}, Mn: {task.fields[mean]:.2g} Tr In: {task.fields[train_inliers]:.2g}'),
                            BarColumn(), TimeRemainingColumn())
            prog_name = log.thread_safe_name('opt')
            opt_task = prog.add_task('opt', total=self.opt.optim.iters)
            prog.update(opt_task, loss=0, lr=0,
                        median=0, mean=0.0, train_inliers=0,
                        advance=0)
            log.map.add(prog_name, prog)
    

        for i in range(self.opt.optim.iters):
            #print('hi')
            optimizer.zero_grad()
            pred = self.graph.depth(pcl_train[:, 1:])
            err = loss_fn(pred, pcl_train[:, :1])
            loss = err.mean()
            loss.backward()
            optimizer.step()
            if isinstance(schedule, ReduceLROnPlateau):
                schedule.step(loss)
            else:
                schedule.step()
            #prog.update(opt_task, advance=1)
            if i % 10 == 0 and pcl_test is not None:
                train_depth = self.depth(pcl_train[:, 1:]).detach()
                train_err = (train_depth - pcl_train[:, :1].cpu()).abs()

                test_depth = self.depth(pcl_test[:, 1:]).detach()
                test_err = (test_depth - pcl_test[:, :1].cpu()).abs()

            
            train_close_inliers = ((pcl_train[close, :1] - to_depth(pred[close])).abs() < 0.05).sum() / close.sum()
            if not quiet:
                prog.update(opt_task, loss=loss.detach().item(), lr=schedule._last_lr[0],
                            median=test_err.median().item(), mean=test_err.mean().item(),
                            train_inliers=train_close_inliers,
                            advance=1)
            
        log.map.remove(prog_name)

    def depth(self, coords):
        x = self.graph.depth(coords.float().to(self.opt.device))
        if self.opt.out.invert_depth:
            depth = 1/x
        else:
            depth = x
        depth = depth.clip(0, MAX_DEPTH)
        return depth.detach().cpu()

    def save_parameters(self, filename):
        g = self.graph.cpu()
        torch.save(g.state_dict(), Path(filename).with_suffix('.pt'))

    def load_parameters(self, filename):
        weights = torch.load(filename)
        self.graph.load_state_dict(weights)
        self.graph.to(self.opt.device)


class Graph(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if self.opt.inpt.type == 'polar':
            self.dim_x = 2
            self.L = opt.pos.freqs
        elif self.opt.inpt.type == 'unit':
            self.dim_x = 3
            self.L = [opt.pos.freqs[0], opt.pos.freqs[0], opt.pos.freqs[1]]
        self.define_network()

    def define_network(self):
        # input layer (default: xyz -> 128)

        if self.opt.pos.use:
            dim_x = 2*sum(self.L) + self.dim_x
        else:
            dim_x = self.dim_x

        if self.opt.out.type == 'scalar':
            dim_out = 1
        else:
            dim_out = 3

        # hidden layers 
        fs = self.opt.layers.nunits
        sizes = [dim_x] + self.opt.layers.n * [fs] + [dim_out]
        self.layers = torch.nn.ModuleList([torch.nn.Linear(i, o)
                                           for i, o in zip(sizes, sizes[1:])])

        # activation functions
        self.activ = torch.nn.functional.relu

    def positional_encoding(self, input): # [B,...,N]
        shape = input.shape
        N = shape[-1]
        enc = []
        for i in range(N):
            freq = 2**torch.arange(self.L[i],dtype=torch.float32,
                                   device=input.device)*np.pi/self.opt.pos.scale # [L]
            spectrum = input[...,i,None]*freq # [B,...,L]
            sin,cos = spectrum.sin(),spectrum.cos() # [B,...,L]
            enc.append(torch.stack([sin,cos],dim=-2).view(*shape[:-1], -1)) # [B,...,2L]
        input_enc = torch.cat(enc, dim=-1).view(*shape[:-1], -1) # [B, ..., 2L_1 + 2L_2 ...]
        return input_enc

    def _forward(self, x): # [B,...,2]
        if self.opt.inpt.type == 'unit':
            x = yp_to_unit(x)
        if self.opt.pos.use:
            x = torch.cat((self.positional_encoding(x), x), dim=-1)
        for layer in self.layers[:-1]:
            x = self.activ(layer(x))
        x = self.layers[-1](x)
        
        return x

    def forward(self, x):
        if len(x) <= EVAL_CHUNK_SIZE:
            return self._forward(x)
        else:
            inds = list(range(0, len(x), EVAL_CHUNK_SIZE)) + [len(x)]
            output = self._forward(x[inds[0]:inds[1]])
            for s,e in list(zip(inds[1:], inds[2:])):
                output = torch.cat((output, self._forward(x[s:e])))
            return output

    def normal(self, coords):
        if self.opt.out.type == 'snorm':
            return self.forward(coords)
        else:
            raise NotImplementedError('Have not implemented normal calculation for scalar depth')

    def depth(self, coords):
        x = self.forward(coords)
        if self.opt.out.type == 'snorm':
            x = (x * yp_to_unit(coords)).sum(dim=-1, keepdims=True)
        return x



