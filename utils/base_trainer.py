from __future__ import division
import sys
import time

import torch
from tqdm import tqdm
tqdm.monitor_interval = 0
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils import CheckpointDataLoader, CheckpointSaver

class BaseTrainer(object):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # override this function to define your model, optimizers etc.
        self.init_fn()
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir)
        lambda1 = lambda epoch: 1#0.3 ** (epoch//30)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']

    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    state_dict = checkpoint[model]
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        if not k.startswith('module.'):
                            name = 'module.'+k
                        else:
                            name = k
                        new_state_dict[name] = v
                    del new_state_dict['module.smpl.betas']
                    del new_state_dict['module.smpl.global_orient']
                    del new_state_dict['module.smpl.body_pose']
                    # load params
                    self.models_dict[model].load_state_dict(new_state_dict, strict=False)
                    print('Checkpoint loaded')
    def train(self):
        """Training process."""
        if self.options.do_test:
            self.step_count = 100
            self.test(0)
            import sys;sys.exit(0)
        torch.manual_seed(0)
        # Run training for num_epochs epochs
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch_count):
            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
            if self.options.do_test:
                self.step_count = 100
                self.test(epoch)
                import sys;sys.exit(0)
            np.random.seed()
            train_data_loader = CheckpointDataLoader(self.train_ds,checkpoint=self.checkpoint,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train,
worker_init_fn=lambda x: np.random.seed())

            # Iterate over all batches in an epoch
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):
                if time.time() < self.endtime:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                    out = self.train_step(batch)
                    self.step_count += 1
                    # Tensorboard logging every summary_steps steps
                    if self.step_count % self.options.summary_steps == 0:
                        self.train_summaries(batch, *out, out_img=self.step_count % (self.options.summary_steps*10)==0)
                    # Save checkpoint every checkpoint_steps steps
                    if self.step_count % self.options.checkpoint_steps == 0:
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step+1, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count)
                        tqdm.write('Checkpoint saved')

                    # Run validation every test_steps steps
                    if self.step_count % self.options.test_steps == 0:
                        self.test(epoch)
                else:
                    tqdm.write('Timeout reached')
                    self.finalize()
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count) 
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)

            self.scheduler.step()

            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None
            # save checkpoint after each epoch
            if (epoch+1) % 10 == 0:
                # self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count)
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size, None, self.step_count)
        return

    # The following methods (with the possible exception of test) have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def train_summaries(self, input_batch):
        raise NotImplementedError('You need to provide a _train_summaries method')

    def test(self, epoch):
        # tqdm.write('start test')
        test_data_loader = CheckpointDataLoader(self.test_ds,checkpoint=None,
                                                 batch_size=self.options.batch_size,
                                                 num_workers=self.options.num_workers,
                                                 pin_memory=self.options.pin_memory,
                                                 shuffle=self.options.shuffle_train,
worker_init_fn=lambda x: np.random.seed())

        # Iterate over all batches in an epoch
        loss = []
        obj = test_data_loader if not self.options.do_test else tqdm(test_data_loader, total=len(self.test_ds) // self.options.batch_size, initial=0)
        for step, batch in enumerate(obj):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
            output, losses = self.train_step(batch, is_train=False)
            loss.append(losses)
            # print(0,losses['losses/cloth_l'])

            # if losses['losses/cloth_l'] > 0.1:
            #     print(batch['dataname'])
            #     print(losses['losses/cloth_l'])
            #     self.train_summaries(batch, output, losses, out_img=self.step_count % (self.options.summary_steps*10)==0, is_train=False)
            #     import sys;sys.exit(0)
            # if step > 100:
            #     break
        final_losses = {}
        for it in loss[0].keys():
            final_losses[it] = np.array([p[it] for p in loss]).mean()
            if self.options.do_test:
                print(it, final_losses[it])
        # Tensorboard logging every summary_steps steps
        self.train_summaries(batch, output, final_losses, out_img=self.step_count % (self.options.summary_steps*10)==0, is_train=False)
        # tqdm.write('end test')
