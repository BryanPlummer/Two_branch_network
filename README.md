# Two-Branch Neural Networks

This repo is from [Two Branch Networks (Liwei Wang, et al.)](https://github.com/lwwang/Two_branch_network) and has been modified to enable testing different settings as done in [this paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Burns_Language_Features_Matter_Effective_Language_Representations_for_Vision-Language_Tasks_ICCV_2019_paper.pdf) like fine-tuning a word embedding, using different language models, and using cca initialized fully connected layers.

## Datasets:

You can download and unpack the caption data using:

  ```Shell
  ./download.sh
  ```

This doesn't include precomputed visual features or language embeddings.  You can obtain the ResNet-152 visual features we used [here](https://drive.google.com/file/d/1Janoli8suKrk9c4MHR0uIYyCK6o7iNCF/view?usp=sharing).  The code is setup to load word embeddings from a space separated text file.  By default the code will load [MT GrOVLE](http://ai.bu.edu/grovle/) embeddings which it assumes has been placed in the `data` directory.  When tuning the `word_embedding_reg` we found values anywhere between 1.5 and 0 to be optimal depending on the word embedding tested, and tuning this parameter for the word embedding can considerably improve performance.

## Usage:

After setting up the datasets, you can train a model using the provided script:

  ```Shell
  ./run_two_branch.sh --train [GPU_ID] [DATASET] [LANGUAGE_MODEL] [EXPERIMENT_NAME]
  # GPU_ID is the GPU you want to test on
  # DATASET in {flickr, coco} determines which dataset is used 
  # LANGUAGE_MODEL in {avg, attend, gru} language encoder used to aggregate word embeddings
  # EXPERIMENT_NAME a descriptor of what to call this experiment
  # Examples:
  ./run_two_branch.sh --train 0 coco avg default_avg
  ./run_two_branch.sh --train 1 flickr gru default_gru
  ```

Training using both `avg` and `attend` language models should take less than an hour on a Titan Xp GPU (on Flickr30K, just a few minutes), but `gru` and other simple alternative recurrent models take considerably longer to train and tends to perform worse on this task when using a pretrained word embedding (see additional results [here]().  More complicated recurrent models may improve performance, however.

Evaluating the model on the 1K test splits for each dataset can be accomplished using:

  ```Shell
  ./run_two_branch.sh --[split] [GPU_ID] [DATASET] [LANGUAGE_MODEL] [CHECKPOINT]
  # GPU_ID is the GPU you want to test on
  # DATASET in {flickr, coco} determines which dataset is used
  # LANGUAGE_MODEL in {avg, attend, gru} language encoder used to aggregate word embeddings
  # CHECKPOINT is the full path of the checkpoint to load
  # Examples:
  ./run_two_branch.sh --test 1 coco attend models/coco/default_attend/two_branch_chpt-22660
  ./run_two_branch.sh --val 0 flickr gru models/flickr/default_gru/two_branch_chpt-5940
  ```

When evaluating it's important to note the discrepancy in the splits on the Flickr30K dataset.  At least two (if not more) splits are used to evaluate the dataset on this task.  The difference in performance between different splits can easily account for a 1-2% difference (this is also true on MSCOCO, but there is more stability in splits there).  It isn't clear if one split always gets better performance than other, and without trying many different models on the same splits it can't be known with any certainty.  We use the same splits as provided by [Flickr30K Entities dataset](http://bryanplummer.com/Flickr30kEntities).  

## Example experiments

Below we provide an example of one of our runs training and testing a self-attention language model using the MT GrOVLE embeddings (which is a little better than the results reported [here](http://openaccess.thecvf.com/content_ICCV_2019/papers/Burns_Language_Features_Matter_Effective_Language_Representations_for_Vision-Language_Tasks_ICCV_2019_paper.pdf), and better than the Two Branch Network's original paper):

  ```Shell
  # The three values for each direction correspond to Recall@{1, 5, 10} (6 numbers total), 
  # and mR refers to the mean of the six recall values.

  # For the Flickr30K dataset
  ./run_two_branch.sh --train 1 flickr attend default_attend
  ./run_two_branch.sh --test 1 flickr attend models/flickr/default_attend/two_branch_chpt-5940

  im2sent: 61.7 86.5 93.2 sent2im: 45.6 76.2 85.3 mr: 74.8

  # For the MSCOCO dataset
  ./run_two_branch.sh --train 2 coco attend default_attend
  ./run_two_branch.sh --test 2 coco attend models/coco/default_attend/two_branch_chpt-22660
  
  im2sent: 68.7 93.5 97.4 sent2im: 54.5 85.6 93.3 mr: 82.2
  ```

This is more comparaible to using the the pca-reduced [HGLMM features](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Klein_Associating_Neural_Word_2015_CVPR_paper.pdf) as reported in [this paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Burns_Language_Features_Matter_Effective_Language_Representations_for_Vision-Language_Tasks_ICCV_2019_paper.pdf).  The HGLMM features may still perform better on this task (after tuning hyperparameters), but is 6K-D rather than 300-D used by the default embeddings, and, thus, requires more parameters and additional computational time.  You can get some precomputed HGLMM features [here](http://ai.bu.edu/grovle/), or can compute them yourself using the code [here](https://github.com/BryanPlummer/pl-clc/tree/master/external/hglmm_fv).  You can also likely improve performance by using [CCA initialization](https://arxiv.org/pdf/1811.07212.pdf) of the fully connected layers, and is supported by this codebase.  It assumes you are provided with a layer weight file in the same format as used by [this repo](https://github.com/BryanPlummer/phrase_detection).

## References

If you use this repo in your project please cite the following papers on the Two Branch Network:

``` markdown
@inproceedings{wang2016learning,
  title={Learning deep structure-preserving image-text embeddings},
  author={Wang, Liwei and Li, Yin and Lazebnik, Svetlana},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5005--5013},
  year={2016}
}

@article{wang2019learning,
  title={Learning two-branch neural networks for image-text matching tasks},
  author={Wang, Liwei and Li, Yin and Huang, Jing and Lazebnik, Svetlana},
  journal={TPAMI},
  volume={41},
  number={2},
  pages={394--407},
  year={2019},
  publisher={IEEE}
}
```


In addition, if you use the MT GrOVLE word embeddings or the self-attention model please also cite:

``` markdown
@InProceedings{burnsLanguage2019,
  title={Language Features Matter: {E}ffective Language Representations for Vision-Language Tasks},
  author={Andrea Burns and Reuben Tan and Kate Saenko and Stan Sclaroff and Bryan A. Plummer},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}
```


Finally, if you use CCA Initialization please cite:

``` markdown
@article{plummerPhrasedetection,
  title={Revisiting Image-Language Networks for Open-ended Phrase Detection},
  author={Bryan A. Plummer and Kevin J. Shih and Yichen Li and Ke Xu and Svetlana Lazebnik and Stan Sclaroff and Kate Saenko},
  journal={arXiv:1811.07212},
  year={2018}
}
``` 

