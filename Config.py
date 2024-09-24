import torch

config_reddit = {
    "dataset": 'reddit',
    "n_head": 4,
    "gat_head": 2,
    "d_model": 128,
    "input_dim": 128,
    "hidden_dim": 128,
    "output_dim": 1,
    "total_nodes": 10000,
    "k_shot": 4,
    "layers": 6,
    "reptile": 1,

    "meta_lr": 0.0015,
    "base_lr": 0.0055,
    "fine_tune_meta_lr": 0.0015,
    "fine_tune_base_lr": 0.0055,
    "fine_tune_lr": 0.015,
    
    
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 64,
    "meta_train_epoch": 10,
    "fine_tune_epoch": 20,
    "train_task_num": 2000,
    "ft_task_num": 20,
    "save_model": True,
    "seed": 123456
}


config_mooc = {
    "dataset": 'mooc',
    "n_head": 2,
    "gat_head": 2,
    "d_model": 32,
    "input_dim": 128,
    "hidden_dim": 128,
    "output_dim": 1,
    "total_nodes": 10000,
    "k_shot": 4,
    "layers": 6,
    "reptile": 1,

    "meta_lr": 0.0008,
    "base_lr": 0.0035,
    "fine_tune_meta_lr": 0.0008,
    "fine_tune_base_lr": 0.0035,
    "fine_tune_lr": 0.008,
    
    
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 64,
    "meta_train_epoch": 10,
    "fine_tune_epoch": 20,
    "train_task_num": 2000,
    "ft_task_num": 20,
    "save_model": True,
    "seed": 123456
}


config_wikipedia = {
    "dataset": 'wikipedia',
    "n_head": 4,
    "gat_head": 2,
    "d_model": 128,
    "input_dim": 128,
    "hidden_dim": 128,
    "output_dim": 1,
    "total_nodes": 10000,
    "k_shot": 2,
    "layers": 6,
    "reptile": 1,

    "meta_lr": 0.0015,
    "base_lr": 0.0055,
    "fine_tune_meta_lr": 0.0027,
    "fine_tune_base_lr": 0.008,
    "fine_tune_lr": 0.027,
    
    
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 64,
    "meta_train_epoch": 10,
    "fine_tune_epoch": 30,
    "train_task_num": 2000,
    "ft_task_num": 20,
    "save_model": True,
    "seed": 123456
}



# this is only for test
config_reddit1 = {
    "dataset": 'reddit1',
    "n_head": 4,
    "gat_head": 2,
    "d_model": 128,
    "input_dim": 128,
    "hidden_dim": 128,
    "output_dim": 1,
    "total_nodes": 10000,
    "k_shot": 2,
    "layers": 10,
    "reptile": 1,

    
    # "meta_lr": 0.002,
    # "base_lr": 0.005,
    # "fine_tune_meta_lr": 0.001,
    # "fine_tune_base_lr": 0.005,
    # "fine_tune_lr": 0.01,
    
    # "meta_lr": 0.0005,
    # "base_lr": 0.002,
    # "fine_tune_meta_lr": 0.0005,
    # "fine_tune_base_lr": 0.002,
    # "fine_tune_lr": 0.01,
    
    
    "meta_lr": 0.0008,
    "base_lr": 0.0042,
    "fine_tune_meta_lr": 0.0008,
    "fine_tune_base_lr": 0.042,
    "fine_tune_lr": 0.008,
    
    
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 64,
    # "meta_train_epoch": 100,
    "meta_train_epoch": 30,
    "fine_tune_epoch": 15,
    "train_task_num": 2000,
    "ft_task_num": 20,
    "save_model": True,
    "seed": 123456
}