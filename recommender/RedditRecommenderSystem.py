import torch
from setfit import SetFitModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import scann
import numpy as np
from sentence_transformers import SentenceTransformer
from simstring.feature_extractor.character_ngram import CharacterNgramFeatureExtractor
from simstring.measure.jaccard import JaccardMeasure
from simstring.database.dict import DictDatabase
from simstring.searcher import Searcher
import pickle
from collections import defaultdict, Counter
import re
import math
import os.path


# Bootstrap the recommender service
class RedditRecommenderSystem:
    def __init__(self, path = '../recommender/data/', with_mentioned_title_exclusion = True):
        self.is_cuda = torch.cuda.is_available()
        self.base_path = path
        self.with_mentioned_title_exclusion = with_mentioned_title_exclusion

    def config(self):
        print(f"Building the recommender. CUDA is {'' if self.is_cuda else 'not'} available.")
        self.movies_recs = self.load_moviedb_data()
        self.movie_id_to_record_map, self.movie_title_to_rec_map = self.map_movies()
        if self.with_mentioned_title_exclusion:
            self.ner, self.title_searcher = self.load_title_extractor()
      
        self.reddit_submissions = self.load_reddit_submissions()
        self.model_submissions,  self.engine_submissions = self.load_model_dependencies(f'{self.base_path}embeddings_trained_subs.pkl',
                                                        'sub_embeddings', 
                                                        f'{self.base_path}model_st_20e/')

        self.model_moviedb, self.engine_moviedb = self.load_model_dependencies(f'{self.base_path}embeddings_mnrl_20e_all.pkl',
                                                        'db_embeddings',
                                                        f'{self.base_path}st_20eps_all_model/')
        
        self.model_setfit, self.setfit_tokenizer = self.load_setfit()

        print('Recommender loading completed.')

    def map_movies(self):
        def get_popularity(x):
            return float(x['popularity']) if 'popularity' in x and not math.isnan(float(x['popularity'])) else 0

        # map results to movie records by title
        movie_id_to_record_map = {m['id']: m for m in self.movies_recs}
        movie_title_to_rec_map = defaultdict(list)
        for m in self.movies_recs:
            movie_title_to_rec_map[self.clean_name(m['title'])].append(m)

        # sort multiple occurances of title by popularity
        for t in movie_title_to_rec_map:
            if len(movie_title_to_rec_map[t]) > 1:
                movie_title_to_rec_map[t] = sorted(movie_title_to_rec_map[t], key= lambda x: get_popularity(x))

        return movie_id_to_record_map, movie_title_to_rec_map

    def load_model_dependencies(self, embeddings_path, embeddings_key, model_name = 'sentence-transformers/all-mpnet-base-v2'):
        if self.is_cuda:
            model = SentenceTransformer(model_name, device='cuda:0') 
        else:
            print(f'Loading {model_name}...')
            model = SentenceTransformer(model_name)

        sub_embeddings = self.load_trained_embeddings(embeddings_path, embeddings_key)
        sub_embeddings_np = np.concatenate(sub_embeddings, axis = 0)
        sub_embeddings_norm = sub_embeddings_np / np.linalg.norm(sub_embeddings_np, axis=1)[:, np.newaxis]

        engine = scann.scann_ops_pybind.builder(sub_embeddings_norm, 1000, "dot_product").tree(
            num_leaves=5000, num_leaves_to_search=500, training_sample_size=250000).score_ah(
            2, anisotropic_quantization_threshold=0.2).reorder(1000).build()

        return model, engine

    def load_trained_embeddings(self, embeddings_path, embeddings_key):
        print(f'Loading {embeddings_key} embeddings...')
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
            embeddings = data[embeddings_key]

            return embeddings

    def load_moviedb_data(self):
        print('Loading movie db (MovieLens) data...')
        with open(f'{self.base_path}tmdb_data.pkl', 'rb') as f:
            data = pickle.load(f)
            movies_recs = data['movies_recs']
            return movies_recs;

    def load_reddit_submissions(self):
        print('Loading Reddit submissions...')
        with open(f'{self.base_path}all_data.pkl', 'rb') as f:
            data = pickle.load(f)
            reddit_submissions_recs = data['reddit_submissions_recs']
            reddit_submissions_recs_2 = data['reddit_submissions_recs_2']
            print('Reddit submissions loaded')

            return reddit_submissions_recs + reddit_submissions_recs_2

    def load_setfit(self):
        dir = f'{self.base_path}setfit_model_final_gpu'
        if not self.is_cuda:
            dir = f'{self.base_path}setfit_model_final_cpu'

        if self.is_cuda:
            print('Loading SetFit for GPU.')
        else:
            print('Loading SetFit for CPU.')
        model = SetFitModel.from_pretrained(dir,
            use_differentiable_head=True,
            head_params={"out_features": 2}
        )
        
        # Load tokenizer of sentence transformers
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2") if self.is_cuda else None
        return model, tokenizer

    def load_title_extractor(self):
        print('Loading Title Extraction pipeline...')
        model_name = f'{self.base_path}bert-large-NER'
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name, local_files_only=True)

        db_path = f'{self.base_path}/title_db.pkl'
        if os.path.isfile(db_path):
            print('Loading inverted index for simstring title search...')
            with open(db_path, 'rb') as f:
                data = pickle.load(f)
                db = data['title_db']
        else:
            print('Creating inverted index for title extraction...')
            movies_titles = [mr['title'] for mr in self.movies_recs]
            db = DictDatabase(CharacterNgramFeatureExtractor(2))
            for a in movies_titles:
                n = self.clean_name(a)
                db.add(n)
            with open(db_path, 'wb') as f:
                pickle.dump({'title_db': db }, f)

        print('Simstring db loaded.')
        searcher = Searcher(db, JaccardMeasure())
        # Define the NER pipeline
        return pipeline("ner", model=model, tokenizer=tokenizer), searcher

    def create_datapoint(self, sent1, sent2, sep_token = '</s>'):
        if self.setfit_tokenizer:
            separator = self.setfit_tokenizer.sep_token
        else:
            separator = sep_token
        return f"""{sent1} {separator} {sent2}"""

    def predict_setfit(self, sent1, sent2):
        datapoint = self.create_datapoint(sent1, sent2)
        v = self.model_setfit.predict(datapoint)
        return v.item()


    def get_body(self, rec):
        text = rec['title']

        if not text.endswith(('.', '?', '!')):
            text += '. '
        text += rec['request']
        return text

    def clean_name(self, name):
        name = name.lower()
        if name.startswith('the '):
            name= name[4:]
        return re.sub(r"\s+", " ", name)

    # pruns inadequate results with the use of setfit model
    def is_fit(self, sent1, sent2):
        return self.predict_setfit(sent1, sent2)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_title(self, title):
        if '__' in title:
                # extract id
                title, id = title.split('__')
                return title, id
        return title, None
    
    def extract_named_entities(self, entities):
        def add_title(title, titles):
            if title and title.lower() != 'the':
                titles.append(title)
                title = None
            return title

        SAME_TITLE_SPACE = 4
        titles = []
        title = None
        for i, entity in enumerate(entities):
            if entity['entity'].startswith("B-"):
                title = add_title(title, titles)
                title = entity['word']
            else:
                if entity['entity'].startswith("I-"):
                    if i == 0 or (i >= 1 and entity['start'] - entities[i- 1]['end'] <= SAME_TITLE_SPACE): #if the next words are close enough they belong to the same title..
                        if title:
                            if entity['word'].startswith('##'):
                                title += entity['word'][2:]
                            else:
                                title += ' ' + entity['word']
                        else:
                            title = add_title(title, titles)
                            title = entity['word'] #..otherwise consider the next word a new title
                else:
                    title = add_title(title, titles)
        if title and title.lower() != 'the':
            titles.append(title)

        print(f'Found entities: {titles}')

        mapped_titles = []
        SIM_THRESH = 0.75
        if len(titles) > 0:
            # map title to nearest real title based on simstring's approximate search
            for title in titles:
                cleaned_title = self.clean_name(title)
                nearest_titles = self.title_searcher.ranked_search(cleaned_title, SIM_THRESH)
                if len(nearest_titles) > 0:
                    print(f'{title} ({cleaned_title}) mapped to {nearest_titles[0]}')
                    mapped_titles.append(nearest_titles[0][1])
        # print(title, res)
        return mapped_titles if len(mapped_titles) > 0 else []


    def exclude_mentioned_titles(self, movie_request, candidates):
        if self.with_mentioned_title_exclusion:
            entites = self.ner(movie_request)
            mentioned_titles = set(self.extract_named_entities(entites))

            filtered_candidates = list(filter(lambda m: self.clean_name(m['title']) not in mentioned_titles, candidates))
            print(f'{len(candidates) - len(filtered_candidates)} candidate(s) filtered out because mentioned in the query.')
            print(f'Filtering out mentioned titles: {mentioned_titles}')

            return filtered_candidates
        return candidates

    def get_candidates(self, movie_request:str, num_cands:int = 20, threshold:float = 0.95):
        """For given user query, produces a set of movie recommendation. This method utilizes efficient vector similarity search (ScaNN)
        on top fine-tuned SentenceTransformers embedding space to find movie submissions from reddit most similar to user request.
        After identifying similar submissions, they are prunned using SetFit to get rid of inadequate requests.
        Among the remaining set, titles extracted from all the replies for all the submissions, a set of titles is ranked and returned.
        Parameters:
        - movie_request: user query
        - num_cands: maximal number of candidates to return
        - threshold: similarity threshold to consider the submissions
        Returns:
        - candidates - set of found movies (from MovieLens dataset)
        - unfit_list - set of submissions that were prunned (for debug purposes)
        """
        DEFAULT_THRES = 0.9
        QUOTA = num_cands
        RES_NUM = 100
        print(f'Search recs for: {movie_request} with threshold {threshold}')

        # embed the user query and search for top best candidates
        req_embedding = self.model_submissions.encode(movie_request) 
        neighbors, distances = self.engine_submissions.search(req_embedding, leaves_to_search=1000, pre_reorder_num_neighbors=RES_NUM)

        # for given similarity threshold get all applicable submissions and extract their titles by rankings
        submissions_ids = [(s, distances[i]) for i, s in  enumerate(neighbors) if distances[i] >= threshold][:QUOTA]
        
        # if there are no submissions by high threshold filter by a second default one
        if len(submissions_ids) == 0:
            threshold = DEFAULT_THRES
            submissions_ids = [(s, distances[i]) for i, s in  enumerate(neighbors) if distances[i] >= threshold][:QUOTA]
        
        # set weighted recommendation rates for each of the submission above threshold
        # this is to ensure we get top recommendations from several submissions if they have close sim scores to request
        dists = [d for d in distances if d >= threshold][:QUOTA]
        dists_weights = self.softmax(dists)

        # to avoid partitioning into single titles from each recommendation (in case of large number of recs) set minimal titles to consider as 2
        rates = [max(2, math.ceil(num_cands * dw)) for dw in dists_weights]
    
        titles_scores = []
        titles = []
        unfit_list = []

        for i, (sub_idx, distance) in enumerate(submissions_ids):
            if len(titles_scores) < num_cands * 3: # take more than final cands number to account for some shuffling of the titles on that list
                if self.is_fit(movie_request, self.get_body(self.reddit_submissions[sub_idx])):

                    ranking = self.reddit_submissions[sub_idx]['ranking']
                    # scores are ranks (how many given number appeared in recommendation comments) times distance - similarity of the request to given recommendation
                    titles_scores.extend([(title, rank * distance) for (title, rank) in ranking[:rates[i]]])
                    titles.extend([title for title,_ in ranking[:rates[i]]])
                else:
                    unfit_list.append((sub_idx, self.reddit_submissions[sub_idx]))

        # sort the results by final rank
        title_counter = Counter(titles)
        # weight titles-scores to give more importance tot titles that repeated accross all the extracted titles from all the comments
        titles_scores = [(title, score * title_counter[title]/len(titles)) for (title, score) in titles_scores]
        titles_scores = sorted(titles_scores, key=lambda x: x[1], reverse=True)
        ids_set = set()
        candidates = []
        for title, _ in titles_scores:   

            title = self.clean_name(title)     

            # extract title and id
            title, id = self.get_title(title)

            movie = None
            if id and id in self.movie_id_to_record_map:
                movie = self.movie_id_to_record_map[id]
            
            if not movie and title in self.movie_title_to_rec_map:
                movie = self.movie_title_to_rec_map[title][0]
                if not id:
                    id = movie['id']

            if movie and id not in ids_set:
                candidates.append(movie)
                ids_set.add(movie['id'])

        # Filter out titles mentioned in the query
        if self.with_mentioned_title_exclusion:
            return self.exclude_mentioned_titles(movie_request, candidates), unfit_list

        return candidates, unfit_list

    
    def get_auxiliary_candidates(self, movie_request, candidates, num_cands = 10, threshold = 0.5):
        """Auxiliary matcher method to aid the set of returned movie recommendations.
        The matcher was trained on pairs of movie description - movie request (all best matching submissions per given movie)
        The aim of auxiliary search is to select movies most closely matching given request recommendation
        in case where the primary search did not identify enough submissions similar to given user query.
        Parameters:
        - movie_request: user query
        - candidates: set of already found movie recommendations
        - num_cands: remaining number of candidates to return
        - threshold: similarity threshold to consider the matching movie descriptions
        Returns:
        - candidates - set of found movies (from MovieLens dataset)
        """
        RES_NUM = 100
        req_embedding = self.model_moviedb.encode(movie_request)
        neighbors, distances = self.engine_moviedb.search(req_embedding, leaves_to_search=1000, pre_reorder_num_neighbors=RES_NUM)
        print(f'Looking for additional {num_cands} MovieLens movies...')
        # for given similarity threshold get all applicable movies
        movies_ids = [(s, distances[i]) for i, s in  enumerate(neighbors) if distances[i] >= threshold]
        # if there are no suitable movies just return empty
        if len(movies_ids) == 0:
            print('No movies found.')
            return []

        print(f'Found {len(movies_ids)} additional candidates.')
        
        movies = []
        i = 0
        cands_ids = set([c['id'] for c in candidates])
        while len(movies) < num_cands and i < len(movies_ids):
            movie = self.movies_recs[movies_ids[i][0]]
            if movie['id'] not in cands_ids:
                movies.append(movie)
            i += 1

        return movies


    def predict_for_request(self, movie_request, num_cands = 20, threshold = 0.968):
        """
        Main recommendation logic to generate a set of movie recommendations for given user query.
        It combines two steps: primary search and auxiliary search (which is only used in case primary search didn't produce enough results)
         Parameters:
        - movie_request: user query
        - num_cands: maximal number of candidates to return
        - threshold: similarity threshold to consider the submissions
        Returns:
        - ids of candidates - set of ids of found movies (from MovieLens dataset)
        - candidates - set of found movies (from MovieLens dataset)
        """
        print(f'Searching candidates for movie request...')
        if not movie_request or type(movie_request) != str:
            raise Exception('wrong movie request type')

       
        candidates, _ = self.get_candidates(movie_request, threshold = threshold, num_cands = num_cands)

        # if less than num_cands candidates is found, add more from auxiliary search
        if len(candidates) < num_cands:   
            additional_candidates = self.get_auxiliary_candidates(movie_request, candidates, num_cands = num_cands - len(candidates), threshold = 0.5)
            if len(additional_candidates) > 0:
                candidates += additional_candidates
                
        return [c['id'] for c in candidates[:num_cands]], candidates
    

