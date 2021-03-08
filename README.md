# I-AID: Identifying Actionable Information from Disaster-related Tweets
This repository contains the implementation of I-AID approach and baseline methods discussed in the paper:
[I-AID: Identifying Actionable Information from Disaster-related Tweets](https://arxiv.org/abs/2008.13544). By ```Hamada M. Zahera, Rricha Jalota, Mohamed A. Sherif and Axel N.Ngnoga (DICE group, Department of Computer Science, Paderborn University)```

This implementation is written in Python 3.6 and uses Tensorflow 2.0

## Abstract: 
Social media plays a significant role in disaster management by providing valuable data about affected people, donations and help requests. Recent studies highlight the need to filter information on social media into fine-grained content categories. However, identifying useful information from massive amounts of social media posts during a crisis is a challenging task. In this paper, we propose I-AID, a multi-model approach to automatically categorize tweets into multi-label types and filter critical information from the enormous volume of social media data.  We use Bidirectional Encoder Representations from Transformers (commonly known as, BERT) to represent tweets into low-dimensional vectors. We thus employ a graph attention network to model the structural information between tweets tokens and their corresponding labels.  We conducted several experiments on two real publicly-available datasets. 
Our results indicate that I-AID outperforms state-of-the-art approaches in terms of weighted-averaged F1-score by $+6\%$ and $+4\%$ on TREC-IS dataset and COVID19-Tweets respectively.

## Citation

## Installtion:
### Dataset:
### Run the code:

## Contact:
If you have any further questions/feedback, please contact corresponding author at [hamada.zahera@uni-paderborn.de](mailto:hamada.zahera@uni-paderborn.de)

