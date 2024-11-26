import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="parses for parameter tuning")
    parser.add_argument("--model", type=int, default=12,)
    parser.add_argument("--dataset", type=int, default=2,)
    parser.add_argument("--pred_dim", type=int, default=128,)
    parser.add_argument("--lr", type=float, default=0.005,)
    parser.add_argument("--lamda", type=float, default=0.02,)
    parser.add_argument("--layer", type=int, default=2,)
    parser.add_argument("--batch", type=int, default=10000,)
    parser.add_argument("--epoch", type=int, default=300,)
    return parser.parse_args()