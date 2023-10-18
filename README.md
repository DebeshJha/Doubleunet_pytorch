# DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation

## Architecture
<p align="center">
<img src="Img/DoubleU-Net.png">
</p>

## Key Contributions:
**Enhanced U-Net Structure:** DoubleU-Net comprises two U-Net structures combined in an innovative manner. This design is intended to capture richer contextual information and deliver better segmentation performance.

**Improved Feature Extraction:** DoubleU-Net employs a VGG19 architecture (pre-trained on ImageNet) as the encoder for the first U-Net. This allows the model to harness the feature extraction capability of VGG19, which is robust for many visual tasks.

**ASPP (Atrous Spatial Pyramid Pooling) Module:** The paper introduces an ASPP module between the two U-Net structures. ASPP has been instrumental in semantic segmentation tasks due to its ability to capture multi-scale information. By incorporating it into the DoubleU-Net, the model is better equipped to handle objects of varying sizes in medical images.

**Performance:** The proposed DoubleU-Net was tested on several medical image segmentation datasets and was found to outperform other state-of-the-art models in many cases.

## Implementation Guidelines:
**Encoder:** Start with a pre-trained VGG19 encoder. This will form the encoder part of the first U-Net.

**U-Net Architecture:** Implement the decoder part for the first U-Net, ensuring to have skip connections from the encoder. Follow this with the ASPP module.

**Second U-Net:** The output from the ASPP module serves as the encoder for the second U-Net. Implement this U-Net ensuring it captures multi-scale features effectively.

**Loss Function:** Depending on the dataset and segmentation task, choose an appropriate loss function. Dice loss or a combination of Dice and cross-entropy loss is commonly used for medical image segmentation.

**Data Augmentation:** Medical datasets can often be small. Augmenting the data can help in training the model better. Consider rotations, flips, and elastic deformations which are common for medical images.

**Evaluation Metrics:** Use appropriate evaluation metrics like Dice coefficient, Jaccard index, or others suitable for the specific segmentation task.



## Citation
Please cite our paper if you find the work useful: 
<pre>
  @INPROCEEDINGS{9183321,
  author={D. {Jha} and M. A. {Riegler} and D. {Johansen} and P. {Halvorsen} and H. D. {Johansen}},
  booktitle={2020 IEEE 33rd International Symposium on Computer-Based Medical Systems (CBMS)}, 
  title={DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation}, 
  year={2020},
  pages={558-564}}
</pre>

## Contact
please contact debesh.jha@northwestern.edu for any further questions. 


