## StyA2K 

This repository is an implementation of the ICCV 2023 paper "All-to-key Attention for Arbitrary Style Transfer". 

### Requirements

+ Ubuntu 18.04
+ Anaconda (Python, Numpy, PIL, etc.)
+ PyTorch 1.9.0
+ torchvision 0.10.0

### Getting Started

  * Inference: 

    * Download [vgg_normalised.pth](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing).

    * The pre-trained models are right in the ./checkpoints/A2K directory, including: latest_net_A2K.pth, latest_net_decoder.pth, and latest_net_transform.pth

    * Configure content_path and style_path in test_A2K.sh to specify the paths to testing content and style images folders, respectively.

    * Run: 

      ```shell
      bash test_A2K.sh
      ```

    * Check the results under the ./results/A2K directory.

  * Train:

    * Download [vgg_normalised.pth](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing).

    * Download [COCO dataset](http://images.cocodataset.org/zips/train2014.zip) and [WikiArt dataset](http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip).

    * Configure content_path, style_path, and image_encoder_path in train_A2K.sh to specify the paths to training content images folders, training style images folders, and "vgg_normalised.pth", respectively.


    * Then, simply run: 

      ```shell
      bash train_A2K.sh
      ```

    * Monitor the training status at http://localhost:8097/. Trained models would be saved in the ./checkpoints/A2k folder.

    * Try other training options in train_A2K.sh. 


### Acknowledgments

  * This code builds heavily on **[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)** and **[AdaAttN](https://github.com/Huage001/AdaAttN)**. Thanks for open-sourcing!
