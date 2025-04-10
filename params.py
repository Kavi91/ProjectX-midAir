import os

class Parameters():
    def __init__(self, batch_size=24):
        self.n_processors = 24
        # Path
        self.data_dir = '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed'
        self.image_dir = self.data_dir
        self.depth_dir = self.data_dir
        self.lidar_dir = self.data_dir
        self.lidar_subdir = 'depth_velo'
        self.pose_dir = self.data_dir
        
        # Mid-Air climate sets
        self.climate_sets = [
            #'Kite_training/cloudy', 'Kite_training/foggy', 'Kite_training/sunny', 'Kite_training/sunset',
            'PLE_training/fall', 'PLE_training/spring', 'PLE_training/winter',
            #'VO_test/foggy', 'VO_test/sunny', 'VO_test/sunset' 
        ]
        
        self.train_traj_ids = {
            #'Kite_training/cloudy': ['trajectory_3000', 'trajectory_3001', 'trajectory_3002'],
            #'Kite_training/foggy':  ['trajectory_2000'],# 'trajectory_2001', 'trajectory_2002'],
            #'Kite_training/sunny':  ['trajectory_0000', 'trajectory_0001', 'trajectory_0002'],
            #'Kite_training/sunset': ['trajectory_1000', 'trajectory_1001', 'trajectory_1002'],
            'PLE_training/spring':  ['trajectory_5000', 'trajectory_5001', 'trajectory_5002'],
            'PLE_training/fall':    ['trajectory_4000', 'trajectory_4001', 'trajectory_4002'],
            'PLE_training/winter':  ['trajectory_6000', 'trajectory_6001', 'trajectory_6002'],
        }

        self.valid_traj_ids = {
            #'Kite_training/cloudy': ['trajectory_3004', 'trajectory_3005'],
            #'Kite_training/foggy':  ['trajectory_2004'],# 'trajectory_2005'],
            #'Kite_training/sunny':  ['trajectory_0004', 'trajectory_0005'],
            #'Kite_training/sunset': ['trajectory_1004', 'trajectory_1005'],
            'PLE_training/spring':  ['trajectory_5003', 'trajectory_5004'],
        }

        self.test_traj_ids = {
            #'Kite_training/foggy':  ['trajectory_2000', 'trajectory_2002'],
            #'Kite_training/cloudy': ['trajectory_3000', 'trajectory_3002'],
            #'Kite_training/sunny':  ['trajectory_0000', 'trajectory_0002'],
            #'Kite_training/sunset': ['trajectory_1000', 'trajectory_1002'],
            'PLE_training/spring':  ['trajectory_5005', 'trajectory_5006'],
        }
        
        self.partition = None
        
        # Data Preprocessing
        self.resize_mode = 'rescale'
        self.img_w = 608  # Reduced from 512
        self.img_h = 184  # Reduced from 384

        self.img_means_03 = (0.5029287934303284, 0.49763786792755127, 0.3706325888633728)
        self.img_stds_03 = (0.255334734916687, 0.23501254618167877, 0.2485671192407608)
        self.img_means_02 = (0.502440869808197, 0.49729645252227783, 0.37063950300216675)
        self.img_stds_02 = (0.2558165192604065, 0.23529765009880066, 0.24865560233592987)
        self.depth_max = 100.0
        self.minus_point_5 = True #flownet requires this

        self.seq_len = (5, 7)
        self.sample_times = 3
        self.overlap = 1

        # Batch size
        self.batch_size = batch_size

        # Data info path
        self.train_data_info_path = 'datainfo/train_df_midair_seq{}x{}_sample{}_b{}.pickle'.format(
            self.seq_len[0], self.seq_len[1], self.sample_times, self.batch_size)
        self.valid_data_info_path = 'datainfo/valid_df_midair_seq{}x{}_sample{}_b{}.pickle'.format(
            self.seq_len[0], self.seq_len[1], self.sample_times, self.batch_size)
        self.test_data_info_path = 'datainfo/test_df_midair_seq{}x{}_sample{}_b{}.pickle'.format(
            self.seq_len[0], self.seq_len[1], self.sample_times, self.batch_size)

        # Model
        self.rnn_hidden_size = 1000
        self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        self.rnn_dropout_out = 0.5
        self.rnn_dropout_between = 0
        self.clip = None
        self.batch_norm = True
        
        # Training
        self.epochs = 600  # Updated to 600 as requested
        self.pin_mem = True
        self.optim = {'opt': 'Adam', 'lr': 1e-4, 'weight_decay': 1e-4}
        
        # Pretrain, Resume training
        self.pretrained_flownet = '/home/krkavinda/DeepVO-pytorch/FlowNet_models/pytorch/flownets_bn_EPE2.459.pth'
        self.resume = False
        self.resume_t_or_v = '.train'

        # Modality flags
        self.enable_rgb = True
        self.enable_depth = False
        self.enable_lidar = False
        self.enable_imu = False
        self.enable_gps = False

        self.gps_loss_weight = 0.5
        self.l2_lambda = 0.00001
        self.k_factor = 100

        # Paths
        self.load_model_path = 'models/midair_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(
            self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
        self.load_optimizer_path = 'models/midair_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format(
            self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]), self.resume_t_or_v)

        self.record_path = 'records/midair_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'.format(
            self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]))
        self.save_model_path = 'models/midair_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(
            self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]))
        self.save_optimzer_path = 'models/midair_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(
            self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]))
        
        # Create directories
        for path in [self.record_path, self.save_model_path, self.save_optimzer_path, self.train_data_info_path, self.test_data_info_path]:
            if not os.path.isdir(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

par = Parameters()
print('Parameters initialized.')