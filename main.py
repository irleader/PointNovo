import torch
import logging
import logging.config
import deepnovo_config
from train_func import train, build_model
from data_reader import DeepNovoDenovoDataset
from model import InferenceModelWrapper
from denovo import IonCNNDenovo
from writer import DenovoWriter
import deepnovo_worker_test
from deepnovo_dia_script_select import find_score_cutoff
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
    if deepnovo_config.FLAGS.train:
        logger.info("training mode")
        train()
    elif deepnovo_config.FLAGS.search_denovo:
        logger.info("denovo mode")
        data_reader = DeepNovoDenovoDataset(feature_filename=deepnovo_config.denovo_input_feature_file,
                                            spectrum_filename=deepnovo_config.denovo_input_spectrum_file)
        denovo_worker = IonCNNDenovo(deepnovo_config.MZ_MAX,
                                     deepnovo_config.knapsack_file,
                                     beam_size=deepnovo_config.FLAGS.beam_size)
        forward_deepnovo, backward_deepnovo, spectrum_cnn = build_model(training=False)
        model_wrapper = InferenceModelWrapper(forward_deepnovo, backward_deepnovo, spectrum_cnn)
        writer = DenovoWriter(deepnovo_config.denovo_output_file)
        denovo_worker.search_denovo(model_wrapper, data_reader, writer)
    elif deepnovo_config.FLAGS.test:
        logger.info("test mode")
        worker_test = deepnovo_worker_test.WorkerTest()
        worker_test.test_accuracy()

        # show 95 accuracy score threshold
        accuracy_cutoff = 0.95
        accuracy_file = deepnovo_config.accuracy_file
        score_cutoff = find_score_cutoff(accuracy_file, accuracy_cutoff)
    else:
        raise RuntimeError("unspecified mode")


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
