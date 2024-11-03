import argparse

def argparser():
    parser = argparse.ArgumentParser(description="Margin Perceptron Argument Parser")
    
    # Add --dataset argument
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset (e.g.,"2d-r16-n10000.txt")')
    
    path_prefix = "datasets/"

    args = parser.parse_args()
    
    # Concatenate the full dataset path
    dataset_path = path_prefix + args.dataset
    return dataset_path

if __name__ == "__main__":
    dataset_path = argparser()
    print(f"Dataset Path: {dataset_path}")