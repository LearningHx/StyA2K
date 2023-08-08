
python train.py \
--content_path ./dataset/train2014 \
--style_path ./dataset/wikiart \
--name A2K \
--model A2K \
--dataset_mode unaligned \
--no_dropout \
--load_size 512 \
--crop_size 256 \
--image_encoder_path checkpoints/vgg_normalised.pth \
--gpu_ids 0 \
--batch_size 8 \
--n_epochs 2 \
--n_epochs_decay 3 \
--display_freq 1 \
--display_port 8097 \
--display_env A2K \
--lambda_no_param_A2K 1.25 \
--lambda_global 10 \
--lambda_content 0 

