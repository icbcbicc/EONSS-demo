import argparse
import TestModel


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument('--img', type=str, default="Dist.jpg", help='name of the image')
    return parser.parse_args()


def main(cfg):
    t = TestModel.Tester(cfg)
    t.eval()


if __name__ == "__main__":
    config = parse_config()
    main(config)
