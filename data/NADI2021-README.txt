================================================================
                    NADI 2021: The Second Nuanced Arabic Dialect Identification Shared Task

=================================================================
Summary
=================================================================

Arabic has a wide variety of dialects, many of which remain under-studied, primarily due to lack of data. The goal of the Nuanced Arabic Dialect Identification (NADI) is to alleviate this bottleneck by affording the community with diverse data from 21 Arab countries. The data can be used for modeling dialects, and NADI focuses on dialect identification. Dialect identification is the task of automatically detecting the source variety of a given text or speech segment. Previous work on Arabic dialect identification has focused on coarse-grained regional varieties such as Gulf or Levantine (e.g., Zaidan and Callison-Burch, 2013; Elfardy and Diab, 2013; Elaraby and Abdul-Mageed, 2018) or country-level varieties (e.g., Bouamor et al., 2018; Zhang and Abdul-Mageed, 2019) such as the MADAR shared task in WANLP 2019 (Bouamor, Hassan, and Habash, 2019). Recently, Abdul-Mageed, Zhang, Elmadany, and Ungar (2020) also developed models for detecting city-level variation. NADI aims at maintaining this theme of modeling fine-grained variation.

=================================================================
Second-NADI-Shared-Task
=================================================================

Description of Data:
--------------------

NADI targets province-level dialects, and as such is the first to focus on naturally-occurring fine-grained dialect at the sub-country level. The NADI 2020 shared task (Abdul-Mageed, Zhang, Bouamor, and Habash, 2020) was held with WANLP 2020. The NADI 2021 shared task will be held with WANLP@EACL2021 and will continue to focus on fine-grained dialects with new datasets and efforts to distinguish both Modern Standard Arabic (MSA) and dialectal Arabic (DA) based on geographical origin. The data covers a total of 100 provinces from all 21 Arab countries and come from the Twitter domain. Evaluation and task set up follow the NADI 2020 shared task. The subtasks involved include:


**Subtask 1**: 
# Subtask 1.1: Country-level MSA identification: A total of 21,000 labeled tweets for training, covering 21 Arab countries.

# Subtask 1.2: Country-level DA identification: A total of 21,000 labeled tweets for training, covering 21 Arab countries.

**Subtask 2**: 
# Subtask 2.1: Province-level MSA identification: A total of 21,000 labeled tweets for training, covering 100 provinces. This is the same dataset as in Subtask 1.1, but with province labels. 

# Subtask 2.1: Province-level DA identification: A total of 21,000 labeled tweets for training, covering 100 provinces. This is the same dataset as in Subtask 1.2, but with province labels. 

**Unlabeled data**: Participants will also be provided with an additional 10M unlabeled tweets that can be used in developing their systems for either or both of the tasks. This is the same unlabeled data as NADI 2020.  

**Labeled Data**
For subtask 1.1 and 2.1, we provide train and development data in two tsv files respectively:
(1) ./Subtask_1.1+2.1_MSA/MSA_train_labeled.tsv
(2) ./Subtask_1.1+2.1_MSA/MSA_dev_labeled.tsv

For subtask 1.2 and 2.2, we provide train and development data are in two tsv files respectively:
(1) ./Subtask_1.2+2.2_DA/DA_train_labeled.tsv
(2) ./Subtask_1.2+2.2_DA/DA_dev_labeled.tsv

These four files have the same structure:

In the *_labeled.tsv files, we provide the following information:

- The first column (#1_tweetid) contains an ID representing the data point/tweet.

- The second column (#2_tweet) contains the content of tweet. We convert user mention and hyperlinks to `USER' and `URL' strings, respectively.

- The third column (#3_country_label) contains gold **country** label of the tweet for Subtask-1.1 or Subtask-1.2.

- The fourth column (#4_province_label) contains gold **province** label of the tweet for Subtask-2.1 or Subtask-2.2.

=================
Sharing NADI Data
=================
Since we are sharing the actual tweets, and as an additional measure to protect Twitter user privacy, we ask that participants not share the distributed data outside their labs nor publish these tweets publicly. Any requests for use of the data outside the shared task can be directed to organizers and will only be accommodated after the shared task submission deadline.

#### File Statistics

Below, we present more details about the contents of the train and development files.

(1) MSA_train_labeled.tsv and DA_train_labeled.tsv contain 21,000 tweets and corresponding labels for country and province, respectively.

(2) MSA_dev_labeled.tsv and DA_dev_labeled.tsv contain 5,000 tweets and corresponding labels for country and province, respectively.

**Unlabeled Data**:
Participants are also provided with an additional 10 million unlabeled tweets that can be used in developing their systems for any of the tasks. 

(1) NADI2021-unlabeled_10M.tsv

In the NADI2021-unlabeled_10M.tsv, we only provide the actual ID of tweets. Participants need to use the Twitter API to collect the texts of these tweets. We provide a script for collecting the tweets in the accompanying distribution (described later in this README).

- The first column (#1 tweet_ID) contains ID of tweet.


Shared Task Metrics and Restrictions:
-------------------------------------
The evaluation metrics will include precision/recall/f-score/accuracy. **Macro Averaged F-score will be the official metric**.

The performance of submitted systems of subtask-1.1 and subtask-1.2 will be evaluated on predictions of country labels for tweets in test set (release later).

The performance of submitted systems of subtask-2.1 and subtask-2.2 will be evaluated on predictions of province labels for tweets in test set (release later).

IMPORTANT: Participants are NOT allowed to use **MSA_dev_labeled.tsv** or **DA_dev_labeled.tsv** for training purposes. Participants must report the performance of their best system on both DEV *and* TEST sets in their Shared Task system description paper.

IMPORTANT: Participants can only use the official TRAIN sets and tweets ***text*** of NADI2021-unlabeled_10M.tsv obtained through "NADI2021-Obtain-Tweets.py". 

Participants are NOT allowed to use any additional tweets, nor are they allowed to use outside information.
Specifically -- participants should not use meta data from Twitter about the users or the tweets, e.g., geo-location data.

External manually labelled data sets are *NOT* allowed.

Shared Task Baselines:
----------------------
Baselines for the various sub-tasks will be
available on the shared task website after TEST data release.
Please check the website and include the respective baseline for each sub-task in your description paper.

-------------------------------------
NADI2021-Twitter-Corpus-License.txt contains the license for using
this data.

Scoring script (NADI2021-DID-Scorer.py) and Submission samples
-------------------------------------
NADI2021-DID-Scorer.py is a python script that will take in two text files containing
true labels and predicted labels and will output accuracy, F1 score, precision, and recall. (Note that the official metric is F1 score).

We provide a sample of gold file and a submission sample file for each sub-task. 

For example:
`subtask11_GOLD.txt' is the gold label file of subtask 1.1. 

The file `UBC_subtask11_dev_1.zip' is the zip file of my first submission.
This zip file contains only one txt file: `UBC_subtask11_dev_1.txt'. 
Unzipping `UBC_subtask11_dev_1.zip', you can get `UBC_subtask11_dev_1.txt' where each line is a label string. 

`subtask11_GOLD.txt' and `UBC_subtask11_dev_1.txt' can be used with the NADI2021-DID-Scorer.py.
NADI2021-DID-Scorer.py evaluate the performance of submssion. 

Please read more description NADI2021-DID-Scorer.py of below.

=================================================================
NADI2021-Obtain-Tweets.py:
=================================================================

NADI2021-Obtain-Tweets.py is a python script that will take in
the NADI2021-unlabeled_10M.tsv file and append a column containing
actual tweet texts corresponding to tweet IDs in the file.

VERY-IMPORTANT (1):  Please make sure to have unicodecsv and tweepy
libraries installed. Make the following calls in terminal:

     pip install unicodecsv
     pip install tweepy

VERY-IMPORTANT (2): Use of this script will require a Twitter
developer's account and corresponding authentication credentials.
The authentication credentials have to be provided in lines
81-84 of the code.

Creating a developer's account and obtaining credentials:
To create a Twitter developer's account, login to Twitter and
go to https://developer.twitter.com/en/apply-for-access.html.
After applying for the developer's account, you will have to wait for the
application to be approved. Twitter may contact you for additional
information about your usage. Please make sure to check your email.

Once the developer's account is created, you will need to create an app.
To create an app, go to apps under the dropdown at your username (upper
right corner). You can now click "create an app."

After an app is created, go to your app details and go to the
"keys and tokens" tab. You will be able to generate following
authentication credentials required for the script:

(1) consumer key
(2) consumer secret
(3) access token
(4) access secret


Usage of the script:

    python3 NADI2021-Obtain-Tweets.py  <input_file> <output_file>

    The input file must be in the same format as the
    NADI2021-unlabeled_10M.tsv

    The output file is the name of the file where the obtained tweets
    will be appended. If any tweet is unavailable, it will write
    <UNAVAILABLE> for that tweet.

Example usage:

    python3 NADI2021-Obtain-Tweets.py NADI2021-unlabeled_10M.tsv NADI2021-10M-full-tweets.tsv

Running the following command will produce the file NADI2021-10M-full-tweets.tsv that
contains obtained tweets appended in the last column along with
the tweet IDs from NADI2021-unlabeled_10M.tsv. 

NOTE: NADI2021-unlabeled_10M.tsv contains 10 millions IDs of tweets. 

=================================================================
/NADI2021-DID-Scorer.py
=================================================================

The scoring script (NADI2021-DID-Scorer.py) is provided at the
root directory of the released data.  NADI2021-DID-Scorer.py
is a python script that will take in two text files containing
true labels and predicted labels and will output accuracy,
F1 score, precision and recall. (Note that the official metric is
F1 score).  The scoring script is used for evaluation of all subtasks.

Please make sure to have sklearn library installed.

Usage of the scorer:

    python3 NADI2021-DID-Scorer.py  <gold-file> <pred-file>

    For verbose mode:

    python3 NADI2021-DID-Scorer.py  <gold-file> <pred-file> -verbose

In the dataset directory, there are example
gold and prediction files. If they are used with the scorer,
they should produce the following results:

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
