import torch
from data_reader import collate_func, DeepNovoTrainDataset
import logging
import logging.config
import deepnovo_config
logger = logging.getLogger(__name__)

def main():
    train_set = DeepNovoTrainDataset(deepnovo_config.input_feature_file_train, deepnovo_config.input_spectrum_file_train)
    data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=deepnovo_config.batch_size,
                                              shuffle=True,
                                              num_workers=deepnovo_config.num_workers,
                                              collate_func=collate_func)
    logger.info("successfully create data loader")

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
