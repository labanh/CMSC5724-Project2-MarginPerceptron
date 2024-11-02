import argparse

def argparser():
    parser = argparse.ArgumentParser(description="Margin Perceptron Argument Parser")
    
    # 添加 --dataset 参数
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Dataset (e.g.,"2d-r16-n10000.txt")')
    
    path_perfix = "datasets/"

    args = parser.parse_args()
    
    # 拼接完整的 dataset 路径
    dataset_path = path_perfix + args.dataset
    return dataset_path

if __name__ == "__main__":
    dataset_path = argparser()
    print(f"Dataset Path: {dataset_path}")