import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import os
import glob
import zipfile
import numpy as np
import pandas as pd
import base64
import datamapplot
import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from transformers import BitsAndBytesConfig, AutoTokenizer, pipeline, AutoModelForCausalLM
from io import BytesIO
from IPython.display import HTML
from tqdm import tqdm
from sentence_transformers import util
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech, VisualRepresentation
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.cluster import BaseCluster
from bertopic.backend import MultiModalBackend
from torch import bfloat16
from key import hf_key
from huggingface_hub import login
from visutils import plot_topic_across_time
tqdm.pandas()
# Auxiliary functions ------------------------------------------------
def compute_and_save_embeddings(embedding_model,approach, umap_model = None, hdbscan_model = None, umap_model_2d = None):
    embeddings = embedding_model.encode(speeches, show_progress_bar=True)
    umap_model = umap_model or UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42, verbose=True)
    hdbscan_model = hdbscan_model or HDBSCAN(min_cluster_size=90, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    umap_model_2d = umap_model_2d or UMAP(n_components=2, n_neighbors=15, random_state=42, metric="cosine", verbose=True)
    reduced_embeddings_2d = umap_model_2d.fit_transform(embeddings)
    clusters = hdbscan_model.fit(reduced_embeddings).labels_
    with open(f'../data/approach_{approach}/5d_embeddings.npy', 'wb') as f:
            np.save(f, reduced_embeddings)
    with open(f'../data/approach_{approach}/2d_embeddings.npy', 'wb') as f:
            np.save(f, reduced_embeddings_2d)
    with open(f'../data/approach_{approach}/clusters.npy', 'wb') as f:
            np.save(f, clusters)
    with open(f'../data/approach_{approach}/embeddings.npy', 'wb') as f:
        np.save(f, embeddings)
    


def set_topic_explainer_pipe():
    login(hf_key)
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type='nf4',  # Normalized float 4
    bnb_4bit_use_double_quant=True,  # Second quantization after the first
    bnb_4bit_compute_dtype=bfloat16  # Computation type
    )
    
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # Llama 2 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Llama 2 Model
    model =AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
    )
    
    model.eval()
    
    # Our text generator
    generator = pipeline(
        model=model, tokenizer=tokenizer,
        task='text-generation',
        temperature=0.1,
        max_new_tokens=500,
        repetition_penalty=1.1
    )
    return generator, tokenizer
    
def set_finetuned_topic_explainer():
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type='nf4',  # Normalized float 4
    bnb_4bit_use_double_quant=True,  # Second quantization after the first
    bnb_4bit_compute_dtype=bfloat16  # Computation type
    )
    model_id = "clibrain/Llama-2-7b-ft-instruct-es"
    
    model =AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    generator = pipeline(
        model=model, tokenizer=tokenizer,
        task='text-generation',
        temperature=0.1,
        max_new_tokens=500,
        repetition_penalty=1.1
    )    
    return generator, tokenizer



    
class Dummy_Dimensionality_Reductor:
  """ Class that simulates a dimensionality reduction process for use pre-computed reduced embeddings on BerTopic """
  def __init__(self, reduced_embeddings):
    self.reduced_embeddings = reduced_embeddings

  def fit(self, X):
    return self

  def transform(self, X):
    return self.reduced_embeddings

def load_np_from_file(path):
     with open(path, 'rb') as f:
         return np.load(f)
def get_word_count(text, lang='spanish'):
    try:
        words = word_tokenize(text, language=lang)
    except:
        words = []
        
    return len(words)

# def split_text(text, n_words = 10, lang = 'spanish'):
#     try:
#         words = word_tokenize(text, language=lang)
#         threshold = min(10
#     except:
#         words = []

#     text = ' '.join(words

spanish_stopwords = ['a','actualmente','adelante','además','afirmó','agregó','ahora','ahí','al','algo','alguna','algunas','alguno','algunos','algún','alrededor','ambos','ampleamos','ante','anterior','antes','apenas','aproximadamente','aquel','aquellas','aquellos','aqui','aquí','arriba','aseguró','así','atras','aunque','ayer','añadió','aún','bajo','bastante','bien','buen','buena','buenas','bueno','buenos','cada','casi','cerca','cierta','ciertas','cierto','ciertos','cinco','comentó','como','con','conocer','conseguimos','conseguir','considera','consideró','consigo','consigue','consiguen','consigues','contra','cosas','creo','cual','cuales','cualquier','cuando','cuanto','cuatro','cuenta','cómo','da','dado','dan','dar','de','debe','deben','debido','decir','dejó','del','demás','dentro','desde','después','dice','dicen','dicho','dieron','diferente','diferentes','dijeron','dijo','dio','donde','dos','durante','e','ejemplo','el','ella','ellas','ello','ellos','embargo','empleais','emplean','emplear','empleas','empleo','en','encima','encuentra','entonces','entre','era','erais','eramos','eran','eras','eres','es','esa','esas','ese','eso','esos','esta','estaba','estabais','estaban','estabas','estad','estada','estadas','estado','estados','estais','estamos','estan','estando','estar','estaremos','estará','estarán','estarás','estaré','estaréis','estaría','estaríais','estaríamos','estarían','estarías','estas','este','estemos','esto','estos','estoy','estuve','estuviera','estuvierais','estuvieran','estuvieras','estuvieron','estuviese','estuvieseis','estuviesen','estuvieses','estuvimos','estuviste','estuvisteis','estuviéramos','estuviésemos','estuvo','está','estábamos','estáis','están','estás','esté','estéis','estén','estés','ex','existe','existen','explicó','expresó','fin','fue','fuera','fuerais','fueran','fueras','fueron','fuese','fueseis','fuesen','fueses','fui','fuimos','fuiste','fuisteis','fuéramos','fuésemos','gran','grandes','gracias', 'gueno','ha','haber','habida','habidas','habido','habidos','habiendo','habremos','habrá','habrán','habrás','habré','habréis','habría','habríais','habríamos','habrían','habrías','habéis','había','habíais','habíamos','habían','habías','hace','haceis','hacemos','hacen','hacer','hacerlo','haces','hacia','haciendo','hago','han','has','hasta','hay','haya','hayamos','hayan','hayas','hayáis','he','hecho','hemos','hicieron','hizo','hoy','hube','hubiera','hubierais','hubieran','hubieras','hubieron','hubiese','hubieseis','hubiesen','hubieses','hubimos','hubiste','hubisteis','hubiéramos','hubiésemos','hubo','igual','incluso','indicó','informó','intenta','intentais','intentamos','intentan','intentar','intentas','intento','ir','junto','la','lado','largo','las','le','les','llegó','lleva','llevar','lo','los','luego','lugar','manera','manifestó','mayor','me','mediante','mejor','mencionó','menos','mi','mientras','mio','mis','misma','mismas','mismo','mismos','modo','momento','mucha','muchas','mucho','muchos','muy','más','mí','mía','mías','mío','míos','nada','nadie','ni','ninguna','ningunas','ninguno','ningunos','ningún','no','nos','nosotras','nosotros','nuestra','nuestras','nuestro','nuestros','nueva','nuevas','nuevo','nuevos','nunca','o','ocho','os','otra','otras','otro','otros','para','parece','parte','partir','pasada','pasado','pero','pesar','poca','pocas','poco','pocos','podeis','podemos','poder','podria','podriais','podriamos','podrian','podrias','podrá','podrán','podría','podrían','poner','por','por qué','porque','posible','primer','primera','primero','primeros','principalmente','propia','propias','propio','propios','próximo','próximos','pudo','pueda','puede','pueden','puedo','pues','que','quedó','queremos','quien','quienes','quiere','quién','qué','realizado','realizar','realizó','respecto','sabe','sabeis','sabemos','saben','saber','sabes','se','sea','seamos','sean','seas','segunda','segundo','según','seis','ser','seremos','será','serán','serás','seré','seréis','sería','seríais','seríamos','serían','serías','seáis','señaló','si','sido','siempre','siendo','siete','sigue','siguiente','sin','sino','sobre','sois','sola','solamente','solas','solo','solos','somos','son','soy','su','sus','suya','suyas','suyo','suyos','sí','sólo','tal','también','tampoco','tan','tanto','te','tendremos','tendrá','tendrán','tendrás','tendré','tendréis','tendría','tendríais','tendríamos','tendrían','tendrías','tened','teneis','tenemos','tener','tenga','tengamos','tengan','tengas','tengo','tengáis','tenida','tenidas','tenido','tenidos','teniendo','tenéis','tenía','teníais','teníamos','tenían','tenías','tercera','ti','tiempo','tiene','tienen','tienes','toda','todas','todavía','todo','todos','total','trabaja','trabajais','trabajamos','trabajan','trabajar','trabajas','trabajo','tras','trata','través','tres','tu','tus','tuve','tuviera','tuvierais','tuvieran','tuvieras','tuvieron','tuviese','tuvieseis','tuviesen','tuvieses','tuvimos','tuviste','tuvisteis','tuviéramos','tuviésemos','tuvo','tuya','tuyas','tuyo','tuyos','tú','ultimo','un','una','unas','uno','unos','usa','usais','usamos','usan','usar','usas','uso','usted','va','vais','valor','vamos','van','varias','varios','vaya','veces','ver','verdad','verdadera','verdadero','vez','vosotras','vosotros','voy','vuestra','vuestras','vuestro','vuestros','y','ya','yo','él','éramos','ésta','éstas','éste','éstos','última','últimas','último','últimos', 'señor', 'tema'] 


def remove_brackets_and_parentheses(text):
    # Define a regular expression pattern to match nested brackets and parentheses
    pattern = r'\([^()]*\)|\[[^\[\]]*\]'
    # Use re.sub() to replace all matches of the pattern with an empty string
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def clean_text(text):
    doc = text
    doc = doc.replace("\n", " ")
    doc = doc.replace("\t", " ")
    doc = doc.replace(" -", " ") 
    doc = doc.replace("inaudible", " ") 
    doc = remove_brackets_and_parentheses(doc)
    doc = ' '.join(doc.split())
    return doc

def sentence_split_nltk(text):
    sentences = nltk.tokenize.sent_tokenize(text, language='spanish')
    return sentences
    

def create_vocabulary(corpus, ignore_words):
    stop_words = ignore_words
    vocabulary = set()
    for phrase in corpus:
        # Tokenize the phrase
        tokens = word_tokenize(phrase.lower())  # Convert to lowercase
        
        bigrams = list(ngrams(tokens, 2))
        
        # Filter out stopwords
        filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        bigrams = [first + ' '+ second for first, second in bigrams if first in filtered_tokens and second in filtered_tokens]
        # Generate bigrams
        # bigrams = list(ngrams(filtered_tokens, 2))
        
        
        # Create vocabulary
        vocabulary = set(filtered_tokens).union(set(bigrams))
    
    return vocabulary



class Text_Topic_Extractor:
    """ Class  that uses BerTopic to find Text clusters and describe them """

    def __init__(self,
                 text_list:list[str],
                 embedding_model:str = "distiluse-base-multilingual-cased-v1",
                 min_topic_size:float  = None,
                 Dimensionality_reductor = None,
                 Cluster_method = None,
                 clusters = None,
                 embeddings = None,
                 reduced_embeddings = None,
                 reduced_embeddings_2d = None,
                 
                ):
        """
        arguments
        ---
            text_list         : list[str]
                list with texts
            embedding_model       : name of embedding transformers model
            min_topic_size    : min size of clusters
            Dimensionality_reductor : Dimensionality Reduction technique. ignored if reduced_embeddings is not None
            Cluster_method: Cluster technique. ignored if clusters is not None
            clusters: list of index identifying the cluster of each text. Can be a path to npy file or a list
            embeddings: precomputed embeddings.Can be a path to npy file or a list
            reduced_embeddings = precomputed reduced embeddings.Can be a path to npy file or a list
            reduced_embeddings_2d = precomputed reduced embeddings 2d used in visualizations.Can be a path to npy file or a list
            
        """
        self.text_list = text_list
        self.embedding_model = SentenceTransformer(embedding_model)
        self.min_topic_size = max(1,int(0.1*len(text_list))) if min_topic_size is None else min_topic_size
        self.hdbscan_model = Cluster_method
        self.embeddings = load_np_from_file(embeddings) if type(embeddings) == str else embeddings
        self.reduced_embeddings = load_np_from_file(reduced_embeddings) if type(reduced_embeddings) == str else reduced_embeddings
        self.reduced_embeddings_2d = load_np_from_file(reduced_embeddings_2d) if type(reduced_embeddings_2d) == str else reduced_embeddings_2d
        self.clusters = load_np_from_file(clusters) if type(clusters) == str else clusters
        self.Dimensionality_reductor = Dimensionality_reductor if reduced_embeddings is None else Dummy_Dimensionality_Reductor(self.reduced_embeddings)
        self.hdbscan_model = Cluster_method if clusters is None else BaseCluster()
        self.topic_info = None
        

    def fit(self, topic_explainer_pipe = False, representation_model = None, top_n_words=10, vectorizer_model = None, ctfidf_model = None):
        """
        arguments
        ----
        topic_explainer_pipe: Hugging face pipeline used to create a readable description of each cluster. If false, No custom name will be added.
        """
        if( representation_model is None):
            representation_model = KeyBERTInspired()
        if (vectorizer_model is None):
            vectorizer_model  = CountVectorizer(stop_words=spanish_stopwords, min_df = 0.1, max_df = 0.8)
        if(ctfidf_model is None):
            ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

        topic_model = BERTopic(

        # Pipeline models
        embedding_model=self.embedding_model,
        umap_model=self.Dimensionality_reductor,
        hdbscan_model=self.hdbscan_model,
        representation_model=representation_model,
        vectorizer_model=vectorizer_model,
        min_topic_size = self.min_topic_size,
        ctfidf_model=ctfidf_model,
        top_n_words=top_n_words,
        verbose=True
    )
        self.topic_model = topic_model.fit(self.text_list, embeddings=self.embeddings, y=self.clusters)
        if(topic_explainer_pipe):
            self.update_labels_LLM(topic_explainer_pipe)
        return


    def describe_topic(self,topic,df, llamapipe, custom_prompt = False):
        """
        Internal function that uses a pipeline to create a readable description of each cluster
        """
        key_words = df[df.Topic == topic].Representation.to_list()[0]
        Representative_Docs = df[df.Topic == topic].Representative_Docs.to_list()[0]
        str_key_words = " "
        for word in key_words:
            str_key_words += f'{word},'
        str_key_words = str_key_words[:-1]

        str_docs = ""
        for docs in Representative_Docs:
            str_docs += f'- {docs}\n'
        str_docs = str_docs[:-1]

        system_prompt = """
        <s>[INST] <<SYS>>
        Eres un asistente servicial, respetuoso y honesto para etiquetar temas que solo responde en español.
        <</SYS>>
        """
        
        example_prompt = """
        Tengo un tema que contiene los siguientes documentos:
- Las dietas tradicionales en la mayoría de las culturas se basaban principalmente en plantas con un poco de carne encima, pero con el aumento de la producción de carne de estilo industrial y las granjas industriales, la carne se ha convertido en un alimento básico.
- La carne, pero especialmente la carne de vacuno, es la palabra alimento en términos de emisiones.
- Comer carne no te convierte en mala persona, no comer carne no te convierte en buena.

El tema se describe con las siguientes palabras clave: 'carne, ternera, comer, comer, emisiones, filete, comida, salud, procesado, pollo'.

Según la información anterior, describa el tema brevemente con un máximo de 5 palabras coherentes. Asegúrate de devolver solo la el tema y nada más.

[/INST] Impactos del consumo de carne
        """
        
        main_prompt = """
        [INST]
        Tengo un tema que contiene los siguientes documentos:
        [DOCUMENTS]
        
        El tema se describe con las siguientes palabras clave: '[KEY-WORDS]'.

Según la información anterior, describa el tema brevemente con un máximo de 5 palabras coherentes. Asegúrate de devolver solo la el tema y nada más.

[/INST]
        """
        main_prompt = main_prompt.replace("[KEY-WORDS]",str_key_words)
        main_prompt = main_prompt.replace("[DOCUMENTS]",str_docs)
    
        prompt = system_prompt + example_prompt + main_prompt 

        res = llamapipe(prompt)
        res = res[0]["generated_text"]
        res = res.split('[/INST]')[2]
        return res

    
    def update_labels_LLM(self,topic_explainer_pipe, prompt = False):
        """
        Internal function that updates the name of each cluster using describe topic function
        """
        df = self.get_topic_info()
        result = df.Topic.progress_apply(lambda topic: self.describe_topic(topic,df,topic_explainer_pipe)).to_list()
        self.topic_model.set_topic_labels(result)
        self.set_topic_info(self.topic_model.get_topic_info())
        return

    def clean_labels(self):
        df = self.get_topic_info()
        result = df.CustomName.progress_apply(clean_text).to_list()
        result = [' '.join(text.split()[:6]) for text in result]
        self.topic_model.set_topic_labels(result)
        self.set_topic_info(self.topic_model.get_topic_info())
        
        
    def get_topic_info(self):
        if(self.topic_info is None):
            self.topic_info = self.topic_model.get_topic_info()
        
        return self.topic_info

    def set_topic_info(self,df):
        self.topic_info = df
        return



    def visualize_2d_clusters(self,hide_document_hover=True, custom_labels=True, hide_annotations = False, top_n  = None):
        if(top_n):
            top_n = range(top_n)
        if(self.reduced_embeddings_2d is None):
            embeddings = self.embedding_model.encode(self.text_list, show_progress_bar=True)
            vis =  self.topic_model.visualize_documents(self.text_list,
                                                    embeddings=embeddings,
                                                    hide_document_hover=hide_document_hover,
                                                   custom_labels = custom_labels,
                                                   hide_annotations = hide_annotations,
                                                       topics = top_n)
        else:
            vis =  self.topic_model.visualize_documents(self.text_list,
                                                    reduced_embeddings=self.reduced_embeddings_2d,
                                                    hide_document_hover=hide_document_hover,
                                                   custom_labels = custom_labels,
                                                   hide_annotations = hide_annotations,
                                                       topics = top_n)
        return vis
    
    def get_document_info(self,df): 
        text_list = df["Document"]
        document_df =self.topic_model.get_document_info(text_list)
        document_df["timestamp"] = df.timestamp
        document_df["speaker"] = df.speaker
        document_df["Old_Index"] = df["index"]
        return document_df

    def save(self,folder_path):
        # Method 1 - safetensors

        self.topic_model.save(folder_path, serialization="safetensors", save_ctfidf=True, save_embedding_model="distiluse-base-multilingual-cased-v1")
        self.topic_info.to_pickle(folder_path+'topic_info.pkl')

    def load(self, topic_model_path, topic_info_path):
        self.topic_model = BERTopic.load(topic_model_path)
        self.topic_info = pd.read_pickle(topic_info_path)

    def create_wordcloud(self, topic):
        text = {word: value for word, value in self.topic_model.get_topic(topic)}
        wc = WordCloud(background_color="white", max_words=1000)
        wc.generate_from_frequencies(text)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    

    def datamap_plot(self, 
                    title = "Text Clusters",
                    subtitle = None,
                    show = True,
                    save_path = None,
                    custom_labels = True):
        if(custom_labels):
            labels = self.topic_model.get_document_info(self.text_list).CustomName.to_list()
        else:
            labels = self.topic_model.get_document_info(self.text_list).Name.to_list()

        if(self.reduced_embeddings_2d is None):
            embeddings = self.embedding_model.encode( self.text_list, show_progress_bar=True)
            reduced_embeddings_2d = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

        else:
            reduced_embeddings_2d = self.reduced_embeddings_2d
        # Run the visualization
        datamapplot.create_plot(
            reduced_embeddings_2d,
            labels,
        
            use_medoids=True,
        
            figsize=(12, 12),
            
            dpi=100,
        
            title=title,
            # Universally set a font family for the plot.
            fontfamily="Roboto",
        
            # Takes a dictionary of keyword arguments that is passed through to
            # matplotlib’s 'title' 'fontdict' arguments.
            title_keywords={
                "fontsize":36,
                "fontfamily":"Roboto Black"
            },
            # Takes a dictionary of keyword arguments that is passed through to
                
            
            # By default DataMapPlot tries to automatically choose a size for the text that will allow
            # all the labels to be laid out well with no overlapping text. The layout algorithm will try
            # to accommodate the size of the text you specify here.
            label_font_size=8,
            label_wrap_width=16,
            label_linespacing=1.25,
            # Default is 1.5. Generally, the values of 1.0 and 2.0 are the extremes.
            # With 1.0 you will have more labels at the top and bottom.
            # With 2.0 you will have more labels on the left and right.
            label_direction_bias=1.3,
            # Controls how large the margin is around the exact bounding box of a label, which is the
            # bounding box used by the algorithm for collision/overlap detection.
            # The default is 1.0, which means the margin is the same size as the label itself.
            # Generally, the fewer labels you have the larger you can make the margin.
            label_margin_factor=2.0,
            # Labels are placed in rings around the core data map. This controls the starting radius for
            # the first ring. Note: you need to provide a radius in data coordinates from the center of the
            # data map.
            # The defaul is selected from the data itself, based on the distance from the center of the
            # most outlying points. Experiment and let the DataMapPlot algoritm try to clean it up.
            label_base_radius=15.0,
        
            # By default anything over 100,000 points uses datashader to create the scatterplot, while
            # plots with fewer points use matplotlib’s scatterplot.
            # If DataMapPlot is using datashader then the point-size should be an integer,
            # say 0, 1, 2, and possibly 3 at most. If however you are matplotlib scatterplot mode then you
            # have a lot more flexibility in the point-size you can use - and in general larger values will
            # be required. Experiment and see what works best.
            point_size=4,
        
            # Market type. There is only support if you are in matplotlib's scatterplot mode.
            # https://matplotlib.org/stable/api/markers_api.html
            marker_type="o",
        
            arrowprops={
                "arrowstyle":"wedge,tail_width=0.5",
                "connectionstyle":"arc3,rad=0.05",
                "linewidth":0,
                "fc":"#33333377"
            },
        
            add_glow=True,
            # Takes a dictionary of keywords that are passed to the 'add_glow_to_scatterplot' function.
            glow_keywords={
                "kernel_bandwidth": 0.75,  # controls how wide the glow spreads.
                "kernel": "cosine",        # controls the kernel type. Default is "gaussian". See https://scikit-learn.org/stable/modules/density.html#kernel-density.
                "n_levels": 32,            # controls how many "levels" there are in the contour plot.
                "max_alpha": 0.9,          # controls the translucency of the glow.
            },
        
            darkmode=False,
        )
        if(show):
            plt.tight_layout()
        if(save_path):
            
            # Save the plot 
            plt.savefig(path)
        return 
