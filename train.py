from configs import get_config
from solver import Solver
from data_loader import get_loader
import sys


if __name__ == '__main__':
    config = get_config(mode='train')
    valid_config = get_config(mode='valid')
    test_config = get_config(mode='test')
    print(config)
    print('video_root_dir = ', config.video_root_dir)
    train_loader = get_loader(config.video_root_dir, config.mode)
    # valid_loader = get_loader(valid_config.video_root_dir, valid_config.mode)
    test_loader  = get_loader(test_config.video_root_dir, test_config.mode)
    solver = Solver(config, train_loader, test_loader)
    solver.build()
    
    if not config.visualize:
        solver.train()
    else:
        solver.visualize()
    # if not config.visualize:
    #     for cross_idx in range(5):
    #         train_loader = get_loader(config.video_root_dir, config.mode, cross_idx)
    #         valid_loader = get_loader(valid_config.video_root_dir, valid_config.mode, cross_idx)
    #         test_loader = get_loader(test_config.video_root_dir, test_config.mode, cross_idx)

    #         solver = Solver(config, train_loader, valid_loader)
    #         solver.build()
    #         solver.train()
    # else:
    #     cross_idx = 0
    #     train_loader = get_loader(config.video_root_dir, config.mode, cross_idx)
    #     test_loader = get_loader(test_config.video_root_dir, test_config.mode, cross_idx)
    #     solver = Solver(config, train_loader, test_loader)
    #     solver.build()
    #     solver.visualize()
