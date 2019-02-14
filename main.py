import torch
from data_reader import collate_func, DeepNovoTrainDataset
import logging
import logging.config
import deepnovo_config
from train_func import train
logger = logging.getLogger(__name__)

def main():
    # train_set = DeepNovoTrainDataset(deepnovo_config.input_feature_file_train, deepnovo_config.input_spectrum_file_train)
    # data_loader = torch.utils.data.DataLoader(dataset=train_set,
    #                                           batch_size=deepnovo_config.batch_size,
    #                                           shuffle=True,
    #                                           num_workers=6,
    #                                           collate_fn=collate_func)
    # for i, temp in enumerate(data_loader):
    #     for a in temp:
    #         print(a.shape)
    #         if len(a.shape) <= 2:
    #             print(a)
    #     break
    # logger.info("successfully create data loader")
    train()

if __name__ == '__main__':
    log_file_name = 'DeepNovo.log'
    d = {
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_name,
                'mode': 'w',
                'formatter': 'standard',
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        }
    }
    logging.config.dictConfig(d)
    main()
