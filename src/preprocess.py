import argparse
from src.utils.preprocess import split_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw", help="Raw dataset folder with cats/ and dogs/")
    parser.add_argument("--processed_dir", default="data/processed", help="Output processed folder")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(args.raw_dir, args.processed_dir, seed=args.seed)
    print(f"Processed dataset created at: {args.processed_dir}")

if __name__ == "__main__":
    main()
