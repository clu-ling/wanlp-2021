================================================================
                    NADI SHARED TASK 2021
          The Second Nuanced Arabic Dialect Identification
=================================================================
Description of TEST phase 
=================================================================
NADI targets province-level dialects, and as such is the first to focus on naturally-occurring fine-grained dialect at the sub-country level. The NADI 2020 shared task (Abdul-Mageed, Zhang, Bouamor, and Habash, 2020) was held with WANLP 2020. The NADI 2021 shared task will be held with WANLP@EACL2021 and will continue to focus on fine-grained dialects with new datasets and efforts to distinguish both Modern Standard Arabic (MSA) and dialectal Arabic (DA) based on geographical origin. The data covers a total of 100 provinces from all 21 Arab countries and come from the Twitter domain. Evaluation and task set up follow the NADI 2020 shared task. The subtasks involved include:

**Subtask 1**: 
# Subtask 1.1: Country-level MSA identification: A total of 21,000 labeled tweets for training, covering 21 Arab countries.

# Subtask 1.2: Country-level DA identification: A total of 21,000 labeled tweets for training, covering 21 Arab countries.

**Subtask 2**: 
# Subtask 2.1: Province-level MSA identification: A total of 21,000 labeled tweets for training, covering 100 provinces. This is the same dataset as in Subtask 1.1, but with province labels. 

# Subtask 2.1: Province-level DA identification: A total of 21,000 labeled tweets for training, covering 100 provinces. This is the same dataset as in Subtask 1.2, but with province labels. 

The evaluation phase of the shared task will be hosted through **CODALAB**. Teams will be provided with a CODALAB link for each shared task.

* CODALAB link for NADI Shared Task Subtask 1.1: https://competitions.codalab.org/competitions/27768
* CODALAB link for NADI Shared Task Subtask 1.2: https://competitions.codalab.org/competitions/27769
* CODALAB link for NADI Shared Task Subtask 2.1: https://competitions.codalab.org/competitions/27770
* CODALAB link for NADI Shared Task Subtask 2.1: https://competitions.codalab.org/competitions/27771

**The deadline of both subtasks are 2021-January-18 23:59:59 (Anywhere in the Earth)**
In the Test phase, each team can submit up to **three system** results for **each subtask**. 

=================================================================
Description of TEST Data:
=================================================================
We provide the unlabeled TEST data for system evaluation. 

For subtask 1.1 and 2.1, we provide unlabeled test data in a tsv file:
(1) ./Subtask_1.1+2.1_MSA/MSA_test_unlabeled.tsv

For subtask 1.2 and 2.2, we provide unlabeled test data are in a tsv file:
(1) ./Subtask_1.2+2.2_DA/DA_test_unlabeled.tsv

These two unlabeled TEST files have the same structure:
'*_test_unlabeled.tsv' file contains hydrated contents of 5,000 tweets. 
You should use the '*_test_unlabeled.tsv' files to predict labels for subtask 1.1/2.1 and subtask 1.2/2.2 using your developed systems, respectively. 

In the *_test_unlabeled.tsv files, we provide the following information:
- The first column (#1_tweetid) contains an ID representing the data point/tweet.

- The second column (#2_tweet) contains the content of tweet. We convert user mention and hyperlinks to `USER' and `URL' strings, respectively.

=================
Sharing NADI Data
=================
Since we are sharing the actual tweets, and as an additional measure to protect Twitter user privacy, we ask that participants not share the distributed data outside their labs nor publish these tweets publicly. Any requests for use of the data outside the shared task can be directed to organizers.

=================================================================
Shared Task Metrics and Restrictions:
=================================================================
The evaluation metrics will include precision/recall/f-score/accuracy. **Macro Averaged F-score will be the official metric**.

The performance of submitted systems of subtask-1.1 and subtask-1.2 will be evaluated on predictions of country labels for tweets in test set.

The performance of submitted systems of subtask-2.1 and subtask-2.2 will be evaluated on predictions of province labels for tweets in test set.

IMPORTANT: Participants are NOT allowed to use **MSA_dev_labeled.tsv** or **DA_dev_labeled.tsv** for training purposes. Participants must report the performance of their best system on both DEV *and* TEST sets in their Shared Task system description paper.

IMPORTANT: Participants can only use the official TRAIN sets and tweets ***text*** of NADI2021-unlabeled_10M.tsv obtained through "NADI2021-Obtain-Tweets.py". 

Participants are NOT allowed to use any additional tweets, nor are they allowed to use outside information.
Specifically -- participants should not use meta data from Twitter about the users or the tweets, e.g., geo-location data.

External manually labelled data sets are *NOT* allowed.

NADI2021-Twitter-Corpus-License.txt contains the license for using this dataset.

=================================================================
Submission Requirments:
=================================================================
You should submit your predication files to the corresponding Codalab competition.

- CODALAB link for NADI Shared Task Subtask 1.1: https://competitions.codalab.org/competitions/27768
- CODALAB link for NADI Shared Task Subtask 1.2: https://competitions.codalab.org/competitions/27769
- CODALAB link for NADI Shared Task Subtask 2.1: https://competitions.codalab.org/competitions/27770
- CODALAB link for NADI Shared Task Subtask 2.1: https://competitions.codalab.org/competitions/27771

You can test your prediction on DEV set (we released before) submitting to the Development phase from the corresponding "development" Codalab tab (accessible when you click the links above).

Then, you should submit your final predictions on the TEST set to the Test phase by the deadline.
**The deadline of both subtasks are 2021-January-18 23:59:59 (Anywhere in the Earth)**
In the Test phase, each team can submit up to **three system** results for **each subtask**. 

The name of your submission should be 'Teamname_Subtask<11/12/21/22>_<dev/test>_NumberOfSubmission.zip' that includes a text file of your prediction. Each zip file only contains one single prediction results. For two examples, my submission 'UBC_subtask11_dev_1.zip' that is the *zip file* of my first prediction for subtask 1.1 on DEV set and contains 'ubc_subtask11_dev_1.txt', and my submission 'ubc_subtask22_test_2.zip' that is the *zip file* of my second prediction for subtask 2.2 on TEST set containing the 'ubc_subtask22_test_2.txt' file.

We provide samples of gold file and a submission sample file for each sub-task. 

For example:
`subtask11_GOLD.txt' is the gold label file of subtask 1.1. 

The file `UBC_subtask11_dev_1.zip' is the zip file of my first submission.
This zip file contains only one txt file: `UBC_subtask11_dev_1.txt'. 
Unzipping `UBC_subtask11_dev_1.zip', you can get `UBC_subtask11_dev_1.txt' where each line is a label string. 
The label in line $x$ in `UBC_subtask11_dev_1.txt' corresponds to the tweet in line $x$ in the DEV set.

`subtask11_GOLD.txt' and `UBC_subtask11_dev_1.txt' can be used with the NADI2021-DID-Scorer.py.
NADI2021-DID-Scorer.py evaluate the performance of submssion. 

Please read more description NADI2021-DID-Scorer.py of below.

=================================================================
Shared Task Baselines:
=================================================================
Baselines for the various sub-tasks is available on the shared task website.
Please check the website and include the respective baseline for each sub-task in your description paper.

=================================================================
./NADI2021-DID-Scorer.py
=================================================================

The scoring script (NADI2021-DID-Scorer.py) is provided at the
root directory of the released data.  NADI2021-DID-Scorer.py
is a python script that will take in two text files containing
true labels and predicted labels and will output accuracy,
F1 score, precision and recall. (Note that the official metric is
F1 score). The scoring script is used for evaluation of all subtasks.

Please make sure to have sklearn library installed.

Usage of the scorer:

    python3 NADI2021-DID-Scorer.py  <gold-file> <pred-file>

    For verbose mode:

    python3 NADI2021-DID-Scorer.py  <gold-file> <pred-file> -verbose

In the dataset directory, there are example
gold and prediction. You should unzip the zip file of submission sample and then run the evaluation command. 
If they are used with the scorer, they should produce the following results:

python3 NADI2021-DID-Scorer.py ./Subtask_1.1+2.1_MSA/subtask11_GOLD.txt ./Subtask_1.1+2.1_MSA/UBC_subtask11_dev_1.txt 

OVERALL SCORES:
MACRO AVERAGE PRECISION SCORE: 5.20 %
MACRO AVERAGE RECALL SCORE: 5.20 %
MACRO AVERAGE F1 SCORE: 5.20 %
OVERALL ACCURACY: 10.60 %

=================================================================
Acknowledgements
=================================================================
The scoring script, the Twitter crawling code are borrowed from the MADAR 2019 Shared Task (Bouamor et al., 2019). The current README and the release package also follows and borrows from the MADAR Shared Task and NADI2020 releases. 

=================================================================
References
=================================================================

@article{zaidan2014arabic,
  title={Arabic dialect identification},
  author={Zaidan, Omar F and Callison-Burch, Chris},
  journal={Computational Linguistics},
  volume={40},
  number={1},
  pages={171--202},
  year={2014},
  publisher={MIT Press}
}

@inproceedings{bouamor2019madar,
  title={The MADAR shared task on Arabic fine-grained dialect identification},
  author={Bouamor, Houda and Hassan, Sabit and Habash, Nizar},
  booktitle={Proceedings of the Fourth Arabic Natural Language Processing Workshop},
  pages={199--207},
  year={2019}
}

@inproceedings{elfardy2013sentence,
  title={Sentence level dialect identification in Arabic},
  author={Elfardy, Heba and Diab, Mona},
  booktitle={Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  pages={456--461},
  year={2013}
}


@inproceedings{bouamor2018madar,
  title={The MADAR Arabic dialect corpus and lexicon},
  author={Bouamor, Houda and Habash, Nizar and Salameh, Mohammad and Zaghouani, Wajdi and Rambow, Owen and Abdulrahim, Dana and Obeid, Ossama and Khalifa, Salam and Eryani, Fadhl and Erdmann, Alexander and others},
  booktitle={Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}


@inproceedings{zhang2019no,
  title={No army, no navy: Bert semi-supervised learning of arabic dialects},
  author={Zhang, Chiyu and Abdul-Mageed, Muhammad},
  booktitle={Proceedings of the Fourth Arabic Natural Language Processing Workshop},
  pages={279--284},
  year={2019}
}

@inproceedings{abdul2018you,
  title={You tweet what you speak: A city-level dataset of arabic dialects},
  author={Abdul-Mageed, Muhammad and Alhuzali, Hassan and Elaraby, Mohamed},
  booktitle={Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}

@inproceedings{elaraby2018deep,
  title={Deep models for arabic dialect identification on benchmarked data},
  author={Elaraby, Mohamed and Abdul-Mageed, Muhammad},
  booktitle={Proceedings of the Fifth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial 2018)},
  pages={263--274},
  year={2018}
}

@inproceedings{mageed-etal-2020-nadi,
    title ={{NADI 2020: The First Nuanced Arabic Dialect Identification Shared Task}},
    author = {Abdul-Mageed, Muhammad and Zhang, Chiyu and Bouamor, Houda and Habash, Nizar},
    booktitle ={{Proceedings of the Fifth Arabic Natural Language Processing Workshop (WANLP 2020)}},
    year = {2020},
    address = {Barcelona, Spain}
}

@inproceedings{abdul-mageed-etal-2020-toward,
    title = "Toward Micro-Dialect Identification in Diaglossic and Code-Switched Environments",
    author = "Abdul-Mageed, Muhammad  and
      Zhang, Chiyu  and
      Elmadany, AbdelRahim  and
      Ungar, Lyle",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    pages = "5855--5876",
}

================================================================
Copyright (c) 2021 The University of British Columbia, Canada; 
Carnegie Mellon University Qatar; New York University Abu Dhabi.
All rights reserved.
================================================================
