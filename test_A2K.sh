
python test.py \
--content_path ./dataset/content \
--style_path ./dataset/style \
--name A2K \
--model A2K \
--dataset_mode unaligned \
--load_size 512 \
--crop_size 512 \
--image_encoder_path checkpoints/vgg_normalised.pth \
--gpu_ids 0 