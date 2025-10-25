from vajra.dataset.split_dota import split_test, split_trainval

# split train and val set, with labels.
split_trainval(
    data_root="/root/data/DOTAv1/",
    save_dir="/root/data/DOTAv1-split/",
    rates=[0.5, 1.0, 1.5],  # multiscale
    gap=500,
)
# split test set, without labels.
split_test(
    data_root="/root/data/DOTAv1/",
    save_dir="/root/data/DOTAv1-split/",
    rates=[0.5, 1.0, 1.5],  # multiscale
    gap=500,
)