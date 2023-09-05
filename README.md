# Introducing Reddit Movie Buff: Your Movie Recommendation Sidekick 🎬

Hey movie buffs! 🍿 Tired of hunting for your next movie night pick? Meet **Reddit Movie Buff** – your AI-powered movie buddy that makes choosing flicks a breeze. Whether you're craving a heart-pounding thriller or a feel-good rom-com, Reddit Movie Buff has got your back.
Think of Reddit Movie Buff as a smart twist on r/MovieSuggestions. It's a cool machine learning project that's all about suggesting movies you'll love – just like the Reddit crew does!

## 🎥 Created by a Movie Lover and Machine Learning Enthusiast

I'm Aleksandra, the movie-loving mind behind Reddit Movie Buff. As a dedicated cinephile with a passion for machine learning solutions, I wanted to create a tool that helps fellow movie enthusiasts find cinematic gems that match their mood and tastes. 


## How It Works

No fuss, no frills. Hop onto 🌐 [Reddit Movie Buff's website](https://lolek27.github.io/reddit-movie-buff/). Type in your movie request, like you're chatting on Reddit. Hit that button and boom! Our AI engine kicks in, whipping up a list of awesome movie suggestions that match your request.

## Behind the Scenes

Our AI engine was trained using tens of thousands of movie requests and recommendations straight from Reddit's [r/MovieSuggestions](https://www.reddit.com/r/MovieSuggestions/) subreddit. That's why it's different than your usual recommendation systems - it doesn't use history of your ratings (it doesn't store anything), neither does it make you filter through genres or creators to get to a movie you want to watch. Because the engine stores the knowledge of the Reddit's movie community, it lets you search for what you'd like to watch like you were asking other reddit users - by simply putting in your request in natural language!

Under the hood, our movie buff is powered by SOTA ML tech stack, including:
- [SentenceTransformers](https://www.sbert.net/)
- [ScaNN](https://github.com/google-research/google-research/tree/master/scann)
- [OpenAI's GPT-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5)
- [SetFit](https://huggingface.co/blog/setfit)
... and more!
  

# Overview of the technical implementation

Reddit Movie Buff system comprises several elements:
1. A website where users enter the query and receive recommendations back
2. REST API service where the request is received and the query is passed to the recommender engine to generate recommendations.
3. Recommender engine, which is where the system generates movie recommendation in two steps:
 3.1. Suggestion Generation
 3.2. Prunning of the results

![architecture drawio](https://github.com/lolek27/reddit-movie-buff-recsys/assets/12550403/f34c224e-3508-4730-b61b-86a5aa22e189)


Before we go over the recommender engine in details, lets understand the underlying data.

## 1. Data
The system rests upon two sets of data:
* [MovieLens dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) - contains metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, release dates etc. This dataset is mainly used to match the extracted titles against the list of existing movies and serve the movies, as well in the Auxilary Matcher (see below)
* Reddit's submissions from [r/MovieSuggestions subreddit](https://www.reddit.com/r/MovieSuggestions/) - almost 40k of submissions from last several years, along with top level answers (called *comments*), collected using reddit's API. This data constitutes the heart of our recommender system. It consists of:
 - submissions - movie requests from reddit's users 
 - comments - corresponding replies from other users containing the movies that best match the requests. Example of submission and it's corresponding comments
 
 (*Note: every example in this documentation comes from r/MovieSuggestions subreddit and contains original spelling, phrasing etc*):


![submission-comments drawio](https://github.com/lolek27/reddit-movie-buff-recsys/assets/12550403/7e303e87-f7d1-4c31-8296-f023df12387c)


 To be able to compare user's query to existing submissions and serve respective movie recommendations we transform the submissions and comments in the following way:
 1. We group all comments which are replies for given submission. We scan through each submission to extract titles (check below section on Title Extraction) from the replies.
 2. We weight each title (recommendation). We use a combination of factors like:
  - how many times given title repeated across all comments
  - what was the score of given comment (number of upvotes for this reply)
  - finally we normalize the weights to be able to compare and combine them with movies from different submission when generating results
3. Factoring in the above results in ranking of movies which we keep along with the corresponding submission

![submission drawio](https://github.com/lolek27/reddit-movie-buff-recsys/assets/12550403/06e1758f-b70c-49d1-8d9a-44d6aaddcce8)

### 1.1. Title Extraction
Extracting titles can be phrased as a NER (Named Entity Recognition) problem and as such is considered difficult because of the free style of reddit's utterences. Consider given examples of comment:
    - *Movies like the machinist, memento or the prestige pretty much a mind bending movies*
    - *Raising Arizona and Con Air are probably his best two IMHO.
    Face/Off and Gone in 60 Seconds are also very fun watches. If you enjoy disaster movies then you might like "Knowing" even though it gets a little tacky.
    I would assume you\'ve watched these before but the National Treasure movies are probably his most enjoyable from start to finish.
    Enjoy!*
    
It's clear that NER systems would struggle to recognize titles where there's no clear interpuction and/or quotes in place. Several langugage models have been compared for this task (including general models like Flan-T5) and finally one of HuggingFace's model: [dslim/bert-large-NER](https://huggingface.co/dslim/bert-large-NER) has been selected. The model tags parts of sentences with tags like 'PER', 'LOC', 'ORG' and 'MISC'. Because of varying style of titles we use most of these tags to assemble titles.
Next we use the title candidates and cross check them against the movie titles data (MovieLens dataset) using approximate string matching like [simstring](https://pypi.org/project/simstring-pure/). The library uses N-gram features of the strings to match them across stored inverted index of all movie titles. 

Only the titles that were correctly mapped to any of the titles are kept for further processing. Among those, an additional step is performed to assure that the found movie actually corresponds to the matter of the submission:

Because NER system doesn't work ideally in the free style realm of reddit's comments, some of the titles are mapped to wrong movies. Thererfore, for each of the movies we queried a general purpose LLM like Flan-T5 and ask if a given submission *S* and corresponding mapped movie *M* have enough in common to consider movie *M* a proper recommendation for *S*. By this means we filtered out some of the movies from the results.


Now let's go over each of the steps of creating movie recommendations in details:

## 2. Generating Set of Recommendations

 Reddit Movie Buff system recommends movies in a semi-collaborative manner. In essence, it searches for movies based on similarity between user's query and the database of known requests. Main steps of the algorithms look as follows: 
 1. Search the dataset of stored user submissions and find submissions most similar to user's query 
 2. Prun the results to keep only the most appropriate submissions
 3. Compose set of recommendation based on similar submissions
 4. Use MovieLens dataset to serve the movie suggestions
 5. Use Auxiliary Matcher to extend the list if necessary

Let's go over each of the steps in details:
### 2.1. Search dataset of stored submissions and find submissions most similar to user's query

We search the dataset of stored submissions to find a set of submissions that are most similar to user's query. In essence, we leverage [SentenceTransformers'](https://www.sbert.net/) dense vectors to embed all the submissions and then use Google's [ScaNN](https://github.com/google-research/google-research/tree/master/scann) - efficient vector search library - to find nearest neighbors for user query.

**SenteceTransfomers (ST)** are SOTA embeddings that use siamese and triplet network structures to derive semantically meaningful sentence embeddings. By mapping both query and the submissions in the embedding space **ScaNN** is able to find submissions similar to user request. However not all candidates foundd by ScaNN convey the same request, that's why we decided to further fine-tune the ST.
Let's explain it with an example. Assume user Y sends the following query:

Y's query: *Recommend me some great korean non-romantic movies. Genre : Action , thriller , suspense anything except romance.*

The search step finds in the space of submissions two requests that seem similar:

Submission A: *Korean Movie Suggestions. Preferably, psychological thriller or wholesome coming of age movies. If you have watched a Korean movie that took you a while to get over it, drop them down below. I’d like to reiterate that I’m looking for movies and not drama series.*,

Submission B: *Please suggest me some great Romantic or Action Korean movies.Hello All! I was introduced to Korean movies back in 2014, and for next couple years I watched some great movies mainly Romance and Action Genre. Below are the movie names that I recollect. I would like to watch more movies that are of similar genre. Sorry for my english. Its not my first language. Any Help appreciated \n- The Man from Nowhere\n- A bittersweet life\n- Oldboy\n- The Terror Live\n- The Wailing\n- The Lake House \n- Always\n- My Sassy Girl*

Looking at these two submissions it's clear that the Submission B contains partially contradictory ask compared to user Y query (Y doesn't want romantic movies, while Submission B does include them). We fine-tune the ST embeddings to improve on the default embedding and make those two requests much less similar. In order to do that we need to prepare a training set of pairs of anchor-positive examples as well negative examples for ST to train on.


 #### 2.1.1. Training set
 The training dataset was created as follows:
 1. For almost 40k submissions (movie recommendation requests) gathered from Reddit's *r/MovieSuggestions* subreddit a subset has been sampled, for which, within the same set a number of *N* nearest neighbours has been selected (with varying threshold of similarity - from 0.99 to 0.1). This step has been accomplished using ScaNN on top of ST's default embeddings of user submissions.
 2. Subsequently, each of the sampled submissions *s* has been coupled with each of the neighbours to create pairs of recommendation requests (s, N*i*), where *i* would range from 1 to 5. By selecting neighbours of vastly different similarity threshold we increased the chance that among the created pairs there would be positive matches as well as negative pairs. By choosing only from the neighbours we also guaranteed that the pairs were hard positives and hard negatives. Around 78k pairs have been created in this step.
 3. Next step was to run the pairs dataset by a reliable, state-fo-the-art LLM and have it label all the pairs as similar and not-similar to create a laballed training set. Two SOTA language models has been considered for this task: 1. **OpenAI's GPT-3.5-turbo** and 2. **Flan-T5-XXL**. 
 4. A set of 120 pairs has been selected from the training set and manually labelled. This set has been used to evaluate the performance of GPT-3.5-turbo and Flan-T5-xxl on the task of comparing two reddit submissions. Two prompts has been created respectively for each of the models and they were asked the same question: if the two users who sent the two submissions can be recommended similar movies or not.
 5. With the use of OpenAI's and Huggingface API's both models have produced their answers. GPT-3.5-turbo's accuracy against the ground-truth labels was about **97%**, while Flan-T5-XXL's accuracy was only **88%**. Given OpenAI's model high performance it was decided that it will be used to labell the whole 78k training set of queries.
 
SentenceTransfomers come with a number of predefined losses (like CosineSimilarityLoss, ContrastiveLoss etc) but we found the underlying  embeddings to produce best results when combined with [MultipleNegativeRankingLoss](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss).
We trained ST for **20 epochs**. Tests run on eval set showed that the set of top 100 nearest neighbours found in the baseline ST embedding space contained **52%** more negative pairs than the fine-tuned set.


### 2.2. Prun the results

The prunning step is designed to eliminate false positives from the generated recommendation set and thus improving on precision. 

The idea of prunning goes as follow: after the search step produces a set of most similar submissions *S* we go through them one by one and get rid of the submissions that do not seem related to user's query. We ask: can user X (who provided submission in *S*) and user Y who sent the query in question be recommended the same titles? If we go back to the examples with Korean movies from 2.1. we see the need of prunning the submission which specifically asks for romantic movies from the set of submissions - the two users cannot be recommended the same movies.

For this step two different binary classificators were introduced, fine-tuned and compared to select better performingon. The selected binary classificator would then make the final call if with the respect to query and given submission.



#### 2.2.1. Binary classificator:

Two binary classificators were fine-tuned and compared for the task of prunning:
 
 A. BERT

 B. SetFit

Below is the overall comparison of two selected classifier - SetFit and BERT (which has been trained twice, on the training set and on augmented training set).
The area under the ROC curve (AUC-ROC) summarizes the performance of each of the classifiers. The performance achieved is as followed (in order):
1. SetFit - AUC-ROC 0.871
2. BERT (augmented) - 0.84
3. BERT - 0.836

A higher AUC-ROC value (closer to 1) suggests better classification performance, while 0.5 indicates random chance (no discrimination). Values above 0.8 are considered to belong to well trained classifiers.

![auc-roc](https://github.com/lolek27/reddit-movie-buff-recsys/assets/12550403/4d6ca697-c88d-4b39-a70d-02299849a435)

 

#### A. BERT
[BERT](https://huggingface.co/docs/transformers/model_doc/bert) is well suited for tasks of comparing two sequences and deciding if they are similar or not. Its [CLS] token is used as a representation of the two sequences, trained for a specific task. During BERT pre-training the [CLS]
token was tuned for the Next Sentence Prediction objective and is
adapted to the request comparison objective during fine-tuning. In the past it has been successfully used for problems like entity matching.

Training procedure:
1. The string representations of two entity descriptions are concatenated using BERT's *\[CLS] Sequence 1 [SEP] Sequence 2 [SEP]* input scheme for sequence classification tasks. The pooled output representation of the [CLS] token from
the last encoder layer is then used as input for a linear classification layer to obtain the final class probabilities, in the case of requests comparison, similar and not-similar.
2. The training set comprises of over **55k pairs of requests (submissions)**. This training set was created from the initial 78k dataset of pairs by balancing it (removing some of the positive class pairs). The fine-tuning started with high recall and gradually built toward higher precision. The best overall metrics were achieved for 6th epoch of training, with:
**accuracy: 0.76**, **F1-score for positive class, precision and recall** all on level of **0.78**.

![bert_ft](https://github.com/lolek27/reddit-movie-buff-recsys/assets/12550403/d3d6020c-3840-41ec-b19e-21c9e543202b)


#### B.SetFit

[SetFit](https://huggingface.co/blog/setfit) is an efficient and prompt-free framework for few-shot fine-tuning of Sentence Transformers. In words of HuggingFace experts:
*SetFit takes advantage of Sentence Transformers’ ability to generate dense embeddings based on paired sentences. In the initial fine-tuning phase stage, it makes use of the limited labeled input data by contrastive training, where positive and negative pairs are created by in-class and out-class selection. The Sentence Transformer model then trains on these pairs (or triplets) and generates dense vectors per example. In the second step, the classification head trains on the encoded embeddings with their respective class labels.*

Training procedure:
1. The embeddings model selected for fine-tuning was 'sentence-transformers/paraphrase-mpnet-base-v2' from HuggingFace.
SetFit comes with a number of predefined losses (like Cosine loss etc) but we found the underlying sentence transformers embeddings to produce best results when combined with a different loss - [MultipleNegativeRankingLoss](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss). For this we needed to extend the SetFit library to enable fine-tuning with MNRL loss.

2. MNRL loss expects one of the two kinds of inputs: either set of anchor-positive pairs (a_1, p_1), (a_2, p_2)… or a set of triplets (a_1, p_1, n_1), (a_2, p_2, n_2) where the third element is a hard negative in addition for the anchor-positive dublet. Trails showed that the triplets with hard negatives provided yielded better results than just the positive pairs. 
Additionally, in MNRL loss the assumption is for every (a_i, p_i) pair every other combination (a_i, p_j) where i!=j is a negative pair. Hence the initial 78k dataset where we had a set of up to 5 paris with every anchor, has been stripped to just include single instances of anchor-positive pair, resulting in a final training set of **13k** pairs.

3. The SetFit training produced a classificator with slightly better results than those of BERT:
**accuracy: 0.79**, **F1-score for positive class: 0.812, Precision: 0.805 and Recall: 0.82**.

![setfit_ft](https://github.com/lolek27/reddit-movie-buff-recsys/assets/12550403/545ae40d-a2fd-44af-9d0c-59c99de09655)

### 2.3. Compose set of recommendations based on similar submissions

After finding set of similar submissions, recommendations are composed as follows:
- consider only submissions above a given similarity threshold. The threshold is the similarity score above which most of the positively matched submissions withing the candidate set fall (it has been set empirically to **95 percentile**) 
- from the ranking lists corresponding to every considered submission take top N recommendations (movie titles). Number of recommendations taken is weighted by the similarity score between the user query and given submission
- Re-rank the movie titles. Each title is ranked based on:
 * how many times given title repeated across the recommendations set
 * distance - similarity of the submission to the user query

### 2.4. Use MovieLens dataset to serve the movie suggestions

Use MovieLens dataset to match against the extracted titles and serve the movie suggestions.

### 2.5. Use Auxiliary Matcher to extend the list if necessary

If there is not enough suggestions generated, use Auxiliary Matcher to extend the list.
Auxiliary Matcher (AM) was created in case user's query doesn't match any submission in our datastore (or the similarity level is below the threshold). AM is another type of search based on SentenceTransformers and ScaNN. In this approach however we do not compare the user's query to submissions, but rather search a movie most similar to the request based on its description and other features.

### Known Limitations
- The movies presented in search results come from Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. This makes latest movie releases absent from the recommendations, also accounting for some occasional unexpected results (for instance presenting older movies titles - sometimes unrelated to the topic - who just happened to share the title with a new release, missing from the movie db).
- The system sometimes produces incorrect (unrelated) recommendations. NER model used to extract titles from submissions and corresponding comments is known to sometimes struggle with the free form style of the utterences. For this reasons some of the extracted titles, and therefore also recommendations based on them, may be wrong.
- The backend runs on AWS' EC2 instance t2.large with 8GB of RAM so it's not a deamon of speed :) It takes up to several seconds to generate results for single request.


## Let's Dive In!

Jump over to [this website](https://lolek27.github.io/reddit-movie-buff/) and let's make your movie nights epic! 🚀🎉

Stay cool and enjoy the show!

*P.S. Popcorn not included.*
