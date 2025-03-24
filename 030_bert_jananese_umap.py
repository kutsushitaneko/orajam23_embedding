import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from adjustText import adjust_text
import umap  # UMAPをインポート

# モデルとトークナイザーのロード
model_name = "tohoku-nlp/bert-base-japanese-v3"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# テキストの例
ikimono_sentences = ["猫は生物です", "犬は生物です", "鳥は生物です", "魚は生物です", "虫は生物です", "象は生物です", "蛇は生物です", "クジラは生物です", "きのこは生物です", "人間は生物です", "生物多様性", "生物兵器は禁止されています", "地球上に最初に現れた生物は？", "火星に生物がいるかどうか研究している"]
namamono_sentences = ["刺身は生物です", "お寿司は生物です", "生牡蠣は生物です", "生卵は生物です", "生ハムは生物です", "生肉は生物です", "生魚は生物です", "生野菜は生物です", "生フルーツは生物です", "生クリームは生物です", "生物には注意が必要です", "生物でお腹を壊した", "生物は加熱した方がいい", "生物は消費期限に注意が必要"]

# ペット文を追加
pet_sentences = ["犬はペットとして人気です", "猫は世界中でペットとして飼われています", "金魚もペットです", "ハムスターはかわいいペットです", "ペットの健康管理は大切です"]
texts = ikimono_sentences + namamono_sentences + pet_sentences

seibutsu_embeddings = []  # 「生物」トークンの埋め込みを文と一緒に保存するリスト
pet_embeddings = []  # 「ペット」トークンの埋め込みを文と一緒に保存するリスト

# テキストリストから各テキストを処理
for i, text in enumerate(texts):
    print(f"\n===== テキスト {i+1}: '{text}' =====")
    
    # トークナイズ
    inputs = tokenizer(text, return_tensors="pt")
    
    # モデルに入力して埋め込みを取得
    outputs = model(**inputs)
    
    # 埋め込みの取得
    embeddings = outputs.last_hidden_state
    
    # トークンインデックスの取得
    token_ids = inputs.input_ids[0]
    token_embeddings = embeddings[0]
    
    # トークンごとの埋め込みを取得
    for token_id, embedding in zip(token_ids, token_embeddings):
        token_text = tokenizer.convert_ids_to_tokens(token_id.item())
        
        # トークンが「生物」の場合のみ処理して保存
        if token_text == "生物":
            print(f"Token ID: {token_id}, トークン: {token_text}, 埋め込み次元数: {embedding.size(0)}")
            
            # 文と埋め込みをセットで保存
            seibutsu_embeddings.append({
                "text": text,
                "embedding": embedding.detach().numpy(),
                "category": "生物"
            })
        
        # トークンが「ペット」の場合も処理して保存
        if token_text == "ペット":
            print(f"Token ID: {token_id}, トークン: {token_text}, 埋め込み次元数: {embedding.size(0)}")
            
            # 文と埋め込みをセットで保存
            pet_embeddings.append({
                "text": text,
                "embedding": embedding.detach().numpy(),
                "category": "ペット"
            })

# 保存したデータの確認
print(f"\n合計 {len(seibutsu_embeddings)} 個の「生物」埋め込みを保存しました")
print(f"合計 {len(pet_embeddings)} 個の「ペット」埋め込みを保存しました")

# すべての埋め込みを結合
all_embeddings = seibutsu_embeddings + pet_embeddings
embeddings_array = np.array([item["embedding"] for item in all_embeddings])

# UMAPで次元削減
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings_array)

# 可視化の準備
plt.figure(figsize=(12, 10))

# 各点のカテゴリを判断し、色を設定
texts_objects = []
for i, item in enumerate(all_embeddings):
    text = item["text"]
    x, y = embeddings_2d[i]
    
    if item["category"] == "ペット":
        color = 'green'
        category_name = 'ペット'
    elif text in ikimono_sentences:
        color = 'red'
        category_name = '生き物'
    else:
        color = 'blue'
        category_name = '生物（なまもの：食べ物）'
    
    plt.scatter(x, y, c=color)
    texts_objects.append(plt.text(x, y, text, fontsize=9))

# テキストの位置を自動調整
adjust_text(texts_objects, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

# グラフの設定
embedding_dim = embeddings_array.shape[1]
plt.title(f'「生物」と「ペット」の埋め込みのUMAP可視化（{embedding_dim}次元から2次元）')

plt.xlabel(f'UMAP次元1')
plt.ylabel(f'UMAP次元2')

# 凡例の追加
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='生き物', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='生物（なまもの：食べ物）', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='ペット', markersize=10)
]
plt.legend(handles=legend_elements)

# グリッド表示
plt.grid(True, linestyle='--', alpha=0.7)

# グラフの表示
plt.tight_layout()

plt.savefig('seibutsu_pet_umap.png')
plt.show()
