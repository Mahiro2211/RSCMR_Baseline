from make_series.make_harma_dataset import make_dir
import json

origin_root = '/home/dhm04/PycharmProjects/RSCR-baseline/dataset/RSITMD/images'

destination_train = '/home/dhm04/PycharmProjects/RSCR-baseline/dataset/AMNFNFinetune/rsitmd/images/train'
destination_test = '/home/dhm04/PycharmProjects/RSCR-baseline/dataset/AMNFNFinetune/rsitmd/images/test'
destination_val = '/home/dhm04/PycharmProjects/RSCR-baseline/dataset/AMNFNFinetune/rsitmd/images/val'

train_json = '/home/dhm04/PycharmProjects/RSCR-baseline/dataset/AMNFNFinetune/ours_train_rsitmd.json'
val_json = '/home/dhm04/PycharmProjects/RSCR-baseline/dataset/AMNFNFinetune/ours_val_rsitmd.json'
test_json = '/home/dhm04/PycharmProjects/RSCR-baseline/dataset/AMNFNFinetune/ours_test_rsitmd.json'

with open(train_json, 'r') as f:
    train_data = json.load(f)
with open(val_json, 'r') as f:
    val_data = json.load(f)
with open(test_json, 'r') as f:
    test_data = json.load(f)

print(train_data[0].keys())
print(train_data[0]['image'])

print(test_data[0].keys())
print(test_data[0]['image'])
make_dir(train_data, origin_root, destination_train)
make_dir(val_data, origin_root, destination_val)
make_dir(test_data, origin_root, destination_test)