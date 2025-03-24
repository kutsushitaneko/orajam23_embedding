from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine as cosine_distance
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.decomposition import PCA
import os
import gensim.downloader as api
from sklearn.preprocessing import normalize
import umap
from adjustText import adjust_text
import warnings
warnings.filterwarnings('ignore')

# モデルのパス設定
ORIGINAL_MODEL_PATH = 'GoogleNews-vectors-negative300.bin'
LIGHT_MODEL_PATH = 'GoogleNews-vectors-negative300.kv'

# モデルのロード関数
def load_word2vec_model():
    # 軽量フォーマットが存在する場合
    if os.path.exists(LIGHT_MODEL_PATH):
        print("軽量フォーマットのモデルを読み込んでいます...")
        return KeyedVectors.load(LIGHT_MODEL_PATH)
    
    # オリジナルのバイナリファイルが存在する場合
    if os.path.exists(ORIGINAL_MODEL_PATH):
        print("オリジナルのモデルを読み込み、軽量フォーマットに変換しています...")
        model = KeyedVectors.load_word2vec_format(ORIGINAL_MODEL_PATH, binary=True)
        model.save(LIGHT_MODEL_PATH)
        return model
    
    # どちらも存在しない場合、ダウンロードを試みる
    print("モデルが見つかりません。word2vec-google-news-300をダウンロードします...")
    try:
        model = api.load('word2vec-google-news-300')
        print("ダウンロードが完了しました。軽量フォーマットに変換して保存します...")
        model.save(LIGHT_MODEL_PATH)
        return model
    except Exception as e:
        raise Exception(f"モデルのダウンロードに失敗しました: {str(e)}")

# モデルのロード
model = load_word2vec_model()

# 単語リスト
words = [
    'king', 'queen', 'prince', 'princess', 
    'man', 'woman', 
    'boy', 'girl', 
    'child', 'adult',    
    'engineer', 'architect', 'doctor', 'nurse', 'pilot',
    'teacher', 'student', 'professor', 'researcher', 'scientist',
    'artist', 'writer', 'musician', 'actor', 'actress',
    'entrepreneur', 'investor', 'entrepreneur',
    'attorney', 'judge', 'police', 'detective', 'spy',
    'cat','lion', 'tiger', 'dog', 'wolf', 'mammal',
    'hawk', 'eagle', 'bird', 
    'ant', 'bee', 'butterfly', 'insects',
    'car', 'truck', 'bicycle', 'motorcycle', 'vehicle', 'machine',
    'house', 'building', 'apartment', 'home', 'room', 'house',
    'tree', 'plant', 'flower', 'grass', 'nature',
    'sky', 'cloud', 'sun', 'moon', 'star',
    'water', 'river', 'sea', 'ocean', 'lake',
    'fire', 'flame', 'smoke', 'fireplace', 'fireworks',
    'joy', 'anger', 'sadness', 'fear', 'surprise',
    'bank', 'money', 'wealth', 'poverty', 'poor',    
]

# 単語ベクトルを辞書に格納
word_vectors = {}
for word in words:
    try:
        word_vectors[word] = model.get_vector(word)
    except KeyError:
        print(f"警告: '{word}' はモデル内に見つかりませんでした。")

# ベクトルリストの準備
vectors = list(word_vectors.values())
labels = list(word_vectors.keys())

# ベクトルの正規化
normalized_vectors = normalize(vectors)

# UMAPの実装
reducer = umap.UMAP(
    n_components=2,
    random_state=42,
    metric='cosine'  # コサイン距離を使用
)
reduced_vectors = reducer.fit_transform(normalized_vectors)

# 可視化
plt.figure(figsize=(12, 10))
for i, label in enumerate(labels):
    plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])

# ラベルの追加（adjust_textを使用）
texts = []
for x, y, label in zip(reduced_vectors[:, 0], reduced_vectors[:, 1], labels):
    texts.append(plt.text(x, y, label))

plt.title('Word2VecのUMAP射影')
plt.xlabel('UMAP次元1')
plt.ylabel('UMAP次元2')
plt.grid(True)

adjust_text(texts,
    arrowprops=dict(arrowstyle='->', color='red', lw=0.5, shrinkA=5),
    expand_points=(1.5, 1.5))

plt.show() 