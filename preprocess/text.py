import pandas as pd
import numpy as np
import torch
import transformers

from transformers import BertTokenizer
from tqdm import tqdm
import torch



################# 文書前処理 #####
def clean_text(raw_text):
    """
    textheroで前処理例
    nltk ライブラリをインストールすることで nltk のデータ利用して stpowordの除去を行える
    """
    import texthero as hero

    raw_text = raw_text

    clean_text = hero.clean(raw_text, pipeline=[
        hero.preprocessing.fillna,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        hero.preprocessing.remove_stopwords
    ])

    # wordcloudで可視化
    hero.visualization.wordcloud(clean_text, colormap='viridis', background_color='white')

    import nltk

    nltk.download('stopwords')
    os.listdir(os.path.expanduser('~/nltk_data/corpora/stopwords/'))

    # 英語とオランダ語を stopword として指定
    custom_stopwords = nltk.corpus.stopwords.words('dutch') + nltk.corpus.stopwords.words('english')

    # 前処理にストップワード除去を加える
    apply_stopword_text = hero.clean(raw_text, pipeline=[
        hero.preprocessing.fillna,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,
        hero.preprocessing.remove_diacritics,
        lambda x: hero.preprocessing.remove_stopwords(x, stopwords=custom_stopwords)
    ])


############## 文書から特徴抽出 #######################
def tfidf(text, tokenizer=None):
    """
    tfidf 適応例
    text: list[text] : トークないずされたテキスト. トークない頭されてない場合は tokenizerを指定するといい
    TODO: きれいにする
    text: 前処理（lowercase, puncituation など) 済みのテキスト
    日本語の場合は tokenize　する関数を tokenizer に指定する.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=100000, tokenizer=tokenizer)),
        ('svd', TruncatedSVD(n_components=50)),
    ])
    z = pipeline.fit_transform(text)


def word2vec_feature():
    """word2vec特徴量を抽出する"""
    import numpy as np
    import pandas as pd

    from pathlib import Path

    from gensim.models import word2vec, KeyedVectors
    from tqdm import tqdm

    mat_col_tec = pd.concat([material, collection, technique], axis=0).reset_index(drop=True)
    mat_col_tec.groupby("object_id")["name"].apply(list)

    # 単語ベクトル表現の次元数
    # 元の語彙数をベースに適当に決めました
    model_size = {
        "material": 20,
        "technique": 8,
        "collection": 3,
        "material_collection": 20,
        "material_technique": 20,
        "collection_technique": 10,
        "material_collection_technique": 25
    }

    n_iter = 100
    w2v_dfs = []
    for df, df_name in zip(
            [
                material, collection, technique,
                mat_col, mat_tec, col_tec, mat_col_tec
            ], [
                "material", "collection", "technique",
                "material_collection",
                "material_technique",
                "collection_technique",
                "material_collection_technique"
            ]):
        df_group = df.groupby("object_id")["name"].apply(list).reset_index()
        # Word2Vecの学習
        w2v_model = word2vec.Word2Vec(df_group["name"].values.tolist(),
                                    size=model_size[df_name],
                                    min_count=1,
                                    window=1,
                                    iter=n_iter)

        # 各文章ごとにそれぞれの単語をベクトル表現に直し、平均をとって文章ベクトルにする
        sentence_vectors = df_group["name"].progress_apply(
            lambda x: np.mean([w2v_model.wv[e] for e in x], axis=0))
        sentence_vectors = np.vstack([x for x in sentence_vectors])
        sentence_vector_df = pd.DataFrame(sentence_vectors,
                                        columns=[f"{df_name}_w2v_{i}"
                                                for i in range(model_size[df_name])])
        sentence_vector_df.index = df_group["object_id"]
        w2v_dfs.append(sentence_vector_df)


class BertSequenceVectorizer:
    """Bert Pretrain を使ってテキストからembeddingを取得
    ex:
    BSV = BertSequenceVectorizer() # インスタンス化します

    train = pd.read_csv(CFG.TRAIN_PATH)
    train['description'] = train['description'].fillna("NaN") # null は代わりのもので埋めます
    train['description_feature'] = train['description'].progress_apply(lambda x: BSV.vectorize(x))
    train[['object_id', 'description', 'description_feature']].head()

    out_df = out_df['description_feature'].apply(pd.Series) #こうするとリストの各要素をDFの行に展開して特徴にできる

    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'bert-base-uncased' #　使用する言語によって変更する
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128 # 文書の最大長。長い文書の場合は増やすなどする。


    def vectorize(self, sentence : str) -> np.array:
        """
        文書のベクトル化
        """
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():    
            return seq_out[0][0].cpu().detach().numpy() # 0番目は [CLS] token, 768 dim の文章特徴量
        else:
            return seq_out[0][0].detach().numpy()

############### 国名変換
# pip install geopy
from geopy.geocoders import Nominatim

def place2country(address):
    """地名を国名へ変換する
    address: str, 変換する地名
    ex:
        place_list = production_place_df['name'].unique()
        for place in place_list:
            try:
                country = place2country(place)
                self.country_dict[place] = country
            except:
                # 国名を取得できない場合は nan
                print('nan place', place)
                self.country_dict[place] = np.nan

    """
    geolocator = Nominatim(user_agent='sample', timeout=200)
    loc = geolocator.geocode(address, language='en')
    coordinates = (loc.latitude, loc.longitude)
    location = geolocator.reverse(coordinates, language='en')
    country = location.raw['address']['country']
    return country

def pred_language():
    """fasttextを使ってテキストの言語を判定する
    !git clone https://github.com/facebookresearch/fastText.git
    !pip install fastText
    !rm -rf fastText
    !wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    """
    from fasttext import load_model

    # パスを書き換えてください
    model = load_model("./lid.176.bin")
    model.predict("Arcadian Landscape with Shepherds and Cattle")
    model.predict("De schilder H.W. Mesdag voor een doek")

    # テキストカラムから言語を取り出し
    lang_list = list(set(input_df[column].fillna("").map(
        lambda x: fs_model.predict(x.replace("\n", ""))[0][0])))
