
from amst_linux.amst_main import apply_amst_transformations
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('raw_folder', type=str)
parser.add_argument('target_folder', type=str)
parser.add_argument('run_name', type=str)

args = parser.parse_args()

if __name__ == '__main__':

    apply_amst_transformations(
        raw_folder=args.raw_folder,
        target_folder=args.target_folder,
        run_name=args.run_name
    )
