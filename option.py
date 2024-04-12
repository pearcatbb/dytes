import argparse

parser = argparse.ArgumentParser(description='Denoise')
parser.add_argument('--dir_data', type=str, default='./dataset')
parser.add_argument('--data_set', type=str, default='zhuanzhang', choices=('xinyongka', 'zhuanzhang'))
parser.add_argument('--bipartite', action='store_true', default=True)
parser.add_argument('--mode', type=str, default='sad', choices=('origin', 'gdn', 'sad')) #模型名
parser.add_argument('--add_scl', action='store_true', default=False)
parser.add_argument('--module_type', type=str, default='graph_attention', choices=('graph_attention', 'graph_sum'))
parser.add_argument('--mask_label', action='store_true', default=False)
parser.add_argument('--mask_ratio', type=float, default=0.01)
# add
parser.add_argument('--dev_alpha', type=float, default=1.0, help="dev loss inlier_loss param")
parser.add_argument('--dev_beta', type=float, default=1.0, help="dev loss outlier_loss param")
parser.add_argument('--anomaly_alpha', type=float, default=1e-1, help="gnn anomaly loss param")
parser.add_argument('--supc_alpha', type=float, default=1e-3, help="gnn supc loss param")
parser.add_argument('--memory_size', type=int, default=5000, help="gdn memory_size")
parser.add_argument('--sample_size', type=int, default=2000, help="gdn sample_size")

##data param
parser.add_argument('--n_neighbors', type=int, default=20, help='Maximum number of connected edge per node')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--num_data_workers', type=int, default=1)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--accelerator', type=str, default='ddp')

##model param
parser.add_argument('--ckpt_file', type=str, default='./')
parser.add_argument('--input_dim', type=int, default=17)
parser.add_argument('--edge_dim', type=int, default=17)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--n_heads', type=int, default=2)
parser.add_argument('--drop_out', type=float, default=0.2)
parser.add_argument('--n_layer', type=int, default=2, help='Number of network layers')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='xinyongka 0.001 zhuanzhang 0.0001')
parser.add_argument('--type_num', type=int, default=1, help='Max type id of node')
# parser.add_argument('--thres', type=float, default=0.1, help='xinyongka 0.3  zhuanzhang 0.5')
parser.add_argument('--model_type', type=str, default='all')
parser.add_argument('--train_ratio', type=float, default=0.5)


args = parser.parse_args()


#xinyongka
# python train.py --data_set xinyongka --learning_rate 1e-3 --train_ratio 0.5 --model_type all

#zhuanzhuang
# python train.py --data_set zhuanzhang --learning_rate 1e-4 --train_ratio 0.5 --model_type all

