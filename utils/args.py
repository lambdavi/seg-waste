import argparse

INF = 9999
def str2tuple(tp=int):

    def convert(s):
        return tuple(tp(i) for i in s.split(','))
    return convert

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--model', type=str, choices=['enet', 'bisenetv2'], default='enet', help='model name')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--bs', type=int, default=8, help='batch size')
    parser.add_argument('--wd', type=float, default=2e-4, help='weight decay')
    parser.add_argument('--m', type=float, default=0.9, help='momentum')

    # New Argument:s

    parser.add_argument('--save', action='store_true', default=False, help='Model saved at the end (training performed)')
    parser.add_argument('--load', action='store_true', default=False, help='Load saved model')
    parser.add_argument('--pred_path', type=str, default=None, help='path of the image to predict')

    return parser
