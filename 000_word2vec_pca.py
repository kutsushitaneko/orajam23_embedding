import gensim.downloader as api
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import japanize_matplotlib
from adjustText import adjust_text
import os
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

# 単語ベクトルの次元数を表示
print(f"単語ベクトルの次元数: {model.vector_size}")

# カテゴリ分け
categories = {
    '人': ['king', 'queen', 'prince', 'princess', 'man', 'woman', 'boy', 'girl', 'child', 'adult',
           'engineer', 'architect', 'doctor', 'nurse', 'pilot', 'teacher', 'student', 'professor', 
           'researcher', 'scientist', 'artist', 'writer', 'musician', 'actor', 'actress',
           'entrepreneur', 'investor', 'attorney', 'judge', 'police', 'detective', 'spy'],
    '動植物': ['cat', 'lion', 'tiger', 'dog', 'wolf', 'mammal', 'hawk', 'eagle', 'bird', 
            'ant', 'bee', 'butterfly', 'insects', 'tree', 'plant', 'flower', 'grass'],
    '自然物': ['nature', 'sky', 'cloud', 'sun', 'moon', 'star', 'water', 'river', 'sea', 'ocean', 
            'lake', 'fire', 'flame', 'smoke', 'bank'],
    '人工物': ['car', 'truck', 'bicycle', 'motorcycle', 'vehicle', 'machine', 'house', 'building', 
            'apartment', 'home', 'room', 'fireplace', 'fireworks'],
    '概念': ['joy', 'anger', 'sadness', 'fear', 'surprise', 'money', 'wealth', 'poverty', 'poor']
}

# 色の定義
category_colors = {
    '人': 'blue',
    '動植物': 'green',
    '自然物': 'red',
    '人工物': 'black',
    '概念': 'purple'
}

# 単語ベクトルを辞書に格納
word_vectors = {}
# categoriesから全ての単語を取得し、重複を排除
words = []
for category, word_list in categories.items():
    for word in word_list:
        if word not in words:  # 重複チェック
            words.append(word)

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

# PCAで次元削減
pca = PCA(n_components=2, random_state=42)
reduced_vectors = pca.fit_transform(normalized_vectors)

# 寄与率の取得
explained_variance_ratio = pca.explained_variance_ratio_

# 可視化（カテゴリごとに色分け）
plt.figure(figsize=(14, 12))

# 各カテゴリごとに点をプロット
for category, color in category_colors.items():
    # カテゴリに含まれる単語のインデックスを取得
    indices = [i for i, label in enumerate(labels) if label in categories[category]]
    
    # そのカテゴリの単語をプロット
    if indices:
        plt.scatter(
            [reduced_vectors[i, 0] for i in indices],
            [reduced_vectors[i, 1] for i in indices],
            color=color,
            label=category,
            s=20,
            alpha=0.8
        )

# ラベルの追加
texts = []
for i, label in enumerate(labels):
    # 単語が属するカテゴリの色を取得
    category = next((cat for cat, words_list in categories.items() if label in words_list), None)
    color = category_colors.get(category, 'black')
    
    # テキストを追加
    texts.append(plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], label, color=color, fontsize=9))

plt.title(f'Word2VecのPCA射影（{model.vector_size}次元から2次元） - カテゴリ別色分け')
plt.xlabel(f'PCA次元1 (寄与率: {explained_variance_ratio[0]:.2%})')
plt.ylabel(f'PCA次元2 (寄与率: {explained_variance_ratio[1]:.2%})')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

# テキスト位置の調整
adjust_text(texts,
    arrowprops=dict(arrowstyle='->', color='red', lw=1.0, alpha=1.0),
        force_points=(0.1, 0.2),
        force_text=(0.5, 1.0),
        expand_points=(1.5, 1.5),
        expand_text=(1.5, 1.5),
        only_move={'points':'xy', 'text':'xy'})

plt.show() 