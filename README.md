We have built our source code on top of 3 repositories:
* DeepWB [1](https://github.com/mahmoudnafifi/Deep_White_Balance)
* mixedillWB [2](https://github.com/mahmoudnafifi/mixedillWB)
* IFRNet [3](https://github.com/birdortyedi/instagram-filter-removal-pytorch)


We have used the synthetic mixed-illuminant evaluation set proposed by Afifi et al. [2]

```
@inproceedings{afifi2020deepWB,
  title={Deep White-Balance Editing},
  author={Afifi, Mahmoud and Brown, Michael S},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}

@inproceedings{afifi2022awb,
  title={Auto White-Balance Correction for Mixed-Illuminant Scenes},
  author={Afifi, Mahmoud and Brubaker, Marcus A. and Brown, Michael S.},
  booktitle={IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2022}
}

@InProceedings{kinli2021ifrnet,
    author={Kinli, Furkan and Ozcan, Baris and Kirac, Furkan},
    title={Instagram Filter Removal on Fashionable Images},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year={2021}
}
```


You can download the synthetic test set (jpg images) from the following links:
* [8-bit JPG images](https://ln4.sync.com/dl/327ce3f30/jd7rvtf6-7tgz43nf-e9ahtm3j-tv8uzxwe)
* [Further information](https://github.com/mahmoudnafifi/mixedillWB#dataset)


To download our trained models:

```
python3 models/download.py 
```

Please do not forget to change -ted (--testing-dir) parameters in bash script with the folder you download the dataset, if you want to simulate.
To simulate the evaluation on the synthetic mixed-illuminant evaluation set -after downloading the dataset and pre-trained weights-:

patch size: 64 & white-balance settings: D S T: 
```
./test_synthetic_64_dst.sh
```

patch size: 128 & white-balance settings: D S T: 
```
./test_synthetic_128_dst.sh
```