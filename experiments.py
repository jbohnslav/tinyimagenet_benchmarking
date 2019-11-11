import argparse
import os
from benchmark_one_epoch import benchmark_one_epoch

def run_experiments(args):

    for imsize in [64, 128, 224, 256]:
        for batch_size in [32, 64, 128, 256, 512]:
            df = benchmark_one_epoch(args.data, 
                    arch=args.arch, 
                    gpu=args.gpu, 
                    batch_size=batch_size, 
                    workers=args.workers,
                    imsize=imsize,
                    notes=args.notes)
    df.to_csv('benchmarks_{}.csv'.format(args.notes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', 
        help='directory with tinyimagenet data')
    parser.add_argument('--arch', type=str, default='resnet50', 
        help='architecture')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
    parser.add_argument('--imsize', type=int, default=224,
                    help='Image size')
    parser.add_argument('-n', '--notes', type=str, default='',
                        help='Notes for a given run. keep short')

    args = parser.parse_args()

    run_experiments(args)