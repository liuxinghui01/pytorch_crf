import argparse
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--corpus_path', type=str, default='./icwb2-data/training/msr_training.utf8', help='训练数据集路径')
parser.add_argument('--run_type', type=str, default='train', help='train模式还是test模式')
parser.add_argument('--train_ratio', type=float, default=0.7, help='train数据占比')
parser.add_argument('--val_ratio', type=float, default=0.2, help='val数据占比')
parser.add_argument('--model_save_dir', type=str, default="", help='模型保存路径')



args = parser.parse_args()
if args.run_type == 'train':
    train(args)