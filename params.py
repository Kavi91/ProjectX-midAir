import os

class Parameters():
    def __init__(self, batch_size=8):
        self.num_workers = 8
        # Base data directory
        self.data_dir = '/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed'
        #self.pose_dir = "/media/krkavinda/New Volume/Mid-Air-Dataset/MidAir_processed/Kite_training/sunny/poses/poses_0008.npy"
        
        # Mid-Air climate sets and trajectory splits
        self.climate_sets = [
            'Kite_training/cloudy', 'Kite_training/foggy', 'Kite_training/sunny', 'Kite_training/sunset',
            'PLE_training/fall', 'PLE_training/spring', 'PLE_training/winter',
        ]
        # Training Trajectories
        self.train_traj_ids = {
            'Kite_training/cloudy': ['trajectory_3000', 'trajectory_3001'], # 'trajectory_3002', 'trajectory_3003'],
            'Kite_training/foggy':  ['trajectory_2000', 'trajectory_2001'], #'trajectory_2002', 'trajectory_2003'],
            'Kite_training/sunny':  ['trajectory_0000', 'trajectory_0001'], #'trajectory_0002', 'trajectory_0003'],
            'Kite_training/sunset': ['trajectory_1000', 'trajectory_1001'], #'trajectory_1002', 'trajectory_1003'],
            'PLE_training/fall':    ['trajectory_4000', 'trajectory_4001'], #'trajectory_4002'],
            'PLE_training/spring':  ['trajectory_5000', 'trajectory_5001'], #'trajectory_5002'],
            'PLE_training/winter':  ['trajectory_6000', 'trajectory_6001'], #'trajectory_6002'],
        }

        # Validation Trajectories
        self.valid_traj_ids = {
            'Kite_training/cloudy': ['trajectory_3006'], #'trajectory_3007'],
            'Kite_training/foggy':  ['trajectory_2008'], #'trajectory_2009'],
            'Kite_training/sunny':  ['trajectory_0010'], #'trajectory_0011'],
            'Kite_training/sunset': ['trajectory_1012'], #'trajectory_1013'],
            'PLE_training/fall':    ['trajectory_4003'], #'trajectory_4004'],
            'PLE_training/spring':  ['trajectory_5003'], #'trajectory_5004'],
            'PLE_training/winter':  ['trajectory_6003'], #'trajectory_6004'],
        }

        # Test Trajectories
        self.test_traj_ids = {
            'Kite_training/sunny': ['trajectory_0008'],  # Reduced for faster testing
            'Kite_training/cloudy': ['trajectory_3004'],
            'Kite_training/foggy': ['trajectory_2006'],
            'Kite_training/sunset': ['trajectory_1008'],
        }
        self.partition = None
        
        # Data Preprocessing settings
        self.resize_mode = 'rescale'
        self.img_w = 608
        self.img_h = 184
        self.img_means_03 = (0.5029287934303284, 0.49763786792755127, 0.3706325888633728)
        self.img_stds_03 = (0.255334734916687, 0.23501254618167877, 0.2485671192407608)
        self.img_means_02 = (0.502440869808197, 0.49729645252227783, 0.37063950300216675)
        self.img_stds_02 = (0.2558165192604065, 0.23529765009880066, 0.24865560233592987)
        self.depth_max = 100.0
        self.minus_point_5 = True
        self.enable_augmentation = False

        self.seq_len = 5
        self.sample_times = 3
        self.overlap = 1

        # Batch size and Data Info file names
        self.batch_size = batch_size
        self.train_data_info_path = 'datainfo/train_df_midair_seq{}_sample{}_b{}.pickle'.format(
            self.seq_len, self.sample_times, self.batch_size)
        self.valid_data_info_path = 'datainfo/valid_df_midair_seq{}_sample{}_b{}.pickle'.format(
            self.seq_len, self.sample_times, self.batch_size)
        self.test_data_info_path = 'datainfo/test_df_midair_seq{}_sample{}_b{}.pickle'.format(
            self.seq_len, self.sample_times, self.batch_size)

        # Model configuration
        self.rnn_hidden_size = 1000
        self.conv_dropout = (0.2,)*8 + (0.5,)
        self.rnn_dropout_out = 0.5 
        self.rnn_dropout_between = 0
        self.clip = None
        self.batch_norm = True
        
        # Training hyperparameters
        self.epochs = 100
        self.pin_mem = True
        self.optim = {'opt': 'Adam', 'lr': 1e-4, 'weight_decay': 1e-5}
        
        # Modality flags
        self.enable_rgb = True
        self.enable_depth = True
        self.enable_lidar = False
        self.enable_imu = True
        self.enable_gps = True

        self.gps_loss_weight = 0.5
        self.l2_lambda = 0
        self.k_factor = 1
        self.depth_gate_scaling = 1.0
        self.imu_gate_scaling = 1.0
        self.translation_loss_weight = 1.0
        self.depth_consistency_loss_weight = 1.0

        # Pretrained model and resume settings
        self.pretrained_flownet = '/home/krkavinda/DeepVO-pytorch/FlowNet_models/pytorch/flownets_bn_EPE2.459.pth'
        self.resume = False
        self.resume_t_or_v = '.train'

        # Model and log file paths
        self.load_model_path = 'models/midair_im{}x{}_s{}_b{}_rnn{}_{}.model{}'.format(
            self.img_h, self.img_w, self.seq_len, self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
        self.load_optimizer_path = 'models/midair_im{}x{}_s{}_b{}_rnn{}_{}.optimizer{}'.format(
            self.img_h, self.img_w, self.seq_len, self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
        self.record_path = 'records/midair_im{}x{}_s{}_b{}_rnn{}_{}.txt'.format(
            self.img_h, self.img_w, self.seq_len, self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]))
        self.save_model_path = 'models/midair_im{}x{}_s{}_b{}_rnn{}_{}'.format(
            self.img_h, self.img_w, self.seq_len, self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]))
        self.save_optimzer_path = 'models/midair_im{}x{}_s{}_b{}_rnn{}_{}'.format(
            self.img_h, self.img_w, self.seq_len, self.batch_size, self.rnn_hidden_size,
            '_'.join([k + str(v) for k, v in self.optim.items()]))

par = Parameters()