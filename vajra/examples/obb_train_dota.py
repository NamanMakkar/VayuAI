from vajra import Vajra

model = Vajra("vajra-v1-small-obb", verbose=True)

model.train(data="DOTAv1.yaml", epochs=200,
            img_size=1024, patience=100, batch=64, dfl=0.75, pretrained=False, val=False,
            save=True, save_period=5, val_period=1,
            project="DOTAv1", name = "vajra_v1_small_obb_train",
            device=[0, 1, 2, 3, 4, 5, 6, 7], workers=8, close_mosaic=0,
            scale=0.9, mixup=0.1, copy_paste=0.0, lr0 = 0.01,
            momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_bias_lr=0.0, warmup_momentum=0.8,
            fliplr=0.5, translate=0.1)
