from utils import TrainOptions
from train import Trainer, AtlasTrainer, AtlasBaselineTrainer, MatTrainer

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    if options.train_module == 'atlas':
        # trainer = AtlasBaselineTrainer(options)
        trainer = AtlasTrainer(options)
    elif options.train_module == 'single':
        trainer = Trainer(options)
    elif options.train_module == 'material':
        trainer = MatTrainer(options)
    else:
        print('module name not accepted!')
        import sys;sys.exit(0)
    trainer.train()
