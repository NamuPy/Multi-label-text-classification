# Multi-label-text-classification
A multi-label text classification is performed using 4 deep learning based model: Word2Vec, Doc2Vec, ELMo and BERT


### Dataset used
- Currently the code is structed assuming input data is from the OPP-115 Corpus (ACL 2016) available at https://usableprivacy.org/data/
-  The OPP-115 Corpus (Online Privacy Policies, set of 115) is a collection of website privacy policies (i.e., in natural language) with annotations that specify data practices in the text
- Each privacy policy was read and annotated by three graduate students in law. 

### Models used to perform classification

1. Word2Vec: A Word2Vec (Skip-Gram) model is trained using genism with all the text data in the complete OPP-115 dataset (only text, no labels), and this is used to extract vector embeddings for each input text. These numerical vector embeddings are further used in a multi-nomial naive bayes model for classification.

2. Doc2Vec: A Doc2Vec (DBOW) model is trained using genism with all the text data in the complete OPP-115 dataset (only text, no labels), and this is used to extract vector embeddings for each input text. These numerical vector embeddings are further used in a multi-nomial naive bayes model for classification.

3. ELMo & BERT: Pretrained models to extract vector embeddings is taken from Tensorflow Hub (Google). For ELMo we use the Tensorflow Hub model version 3 (Peters, M. E. et al. 2018) and BERT ‘bert_uncased_L-12_H-768_A-12’ model and its version 1 (Devlin, J. et al., 2018). The extracted vector embeddings sizes of 1024 and 768 respectively from pretrained ELMo and BERT are used, over which a dense layer of size 512 with relu activation is added and a final output layer of size 12 for the 12 OPP-115 labels, with a softmax activation function. We use categorical crossentropy loss along with the adam optimizer to train for the multi-label classification. 


### Train-Test Split
The experimentation with the OPP-115 dataset includes a 10-fold cross validation, where in each fold the data is split into train and test set into 75 train websites and 40 test websites. The split across 10 iterations is performed such that each website uniformly falls in test set at most 4 times, and for this uniform split a custom algorithm has been coded.


### Results and comparison across models in terms of F-1 scores
The best performing model is BERT after upsampling of training dataset, where resulting average F-1 score across 12 labels is 0.72, which is much higher than the benchmark score of 0.61 mentioned in paper by Shomir Wilson et. al 2016 using SVM. For detailed results across all models contact nmc5751@psu.edu


### References

Wilson, S., Schaub, F., Dara, A. A., Liu, F., Cherivirala, S., Leon, P. G., ... & Norton, T. B. (2016, August). The creation and analysis of a website privacy policy corpus. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 1330-1340).

Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep contextualized word representations. arXiv preprint arXiv:1802.05365.

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.




