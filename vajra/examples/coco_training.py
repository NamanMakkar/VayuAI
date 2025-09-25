from vajra import Vajra

model = Vajra("vajra-v1-xlarge-det", verbose=True)

model.train(data="coco.yaml", epochs=600,
            img_size=640, patience=100, batch=128,
            save=True, save_period=5, val_period=1,
            project="COCO", name = "vajra_v1_xlarge_det_train",
            device=[0, 1, 2, 3, 4, 5, 6, 7], workers=8,
            scale=0.9, mixup=0.2, copy_paste=0.6, lr0 = 0.01,
            momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_bias_lr=0.0, warmup_momentum=0.8,
            fliplr=0.5, translate=0.1)