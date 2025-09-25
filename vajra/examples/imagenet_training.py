from vajra import Vajra

model = Vajra("vajra-v1-large-cls", verbose=True)
model.train(data="../data/imagenet", device=0, epochs=200, batch=256, img_size=224, save=True, save_period=5, hsv_s = 0.4, erasing=0.4, lr0=0.2, weight_decay=1e-4, warmup_epochs=0, momentum=0.9, cos_lr=True, optimizer="SGD", project="Imagenet_train", name="vajra-v1-large-cls")