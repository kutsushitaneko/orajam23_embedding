import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns

# モデルとトークナイザーのロード
model_name = "tohoku-nlp/bert-base-japanese-v3"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# テキストの例
ikimono_sentences = ["猫は生物です", "犬は生物です", "鳥は生物です", "魚は生物です", "虫は生物です", "象は生物です", "蛇は生物です", "クジラは生物です", "きのこは生物です", "人間は生物です", "生物多様性", "生物兵器は禁止されています", "地球上に最初に現れた生物は？", "火星に生物がいるかどうか研究している"]
namamono_sentences = ["刺身は生物です", "お寿司は生物です", "生牡蠣は生物です", "生卵は生物です", "生ハムは生物です", "生肉は生物です", "生魚は生物です", "生野菜は生物です", "生フルーツは生物です", "生クリームは生物です", "生物には注意が必要です", "生物でお腹を壊した", "生物は加熱した方がいい", "生物は消費期限に注意が必要"]

# ペット文を追加
pet_sentences = ["猫は世界中でペットとして飼われています"]
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

# 類似度計算（コサイン類似度）
print("\n===== 埋め込みの類似度計算 =====")

# 埋め込みをnumpy配列に変換
seibutsu_embeddings_array = np.array([item["embedding"] for item in seibutsu_embeddings])
pet_embeddings_array = np.array([item["embedding"] for item in pet_embeddings])

# ペットと生物の間の類似度行列を計算
similarity_matrix = cosine_similarity(pet_embeddings_array, seibutsu_embeddings_array)

# テキストとカテゴリの表示用リスト
pet_texts = [item["text"] for item in pet_embeddings]
seibutsu_texts = [item["text"] for item in seibutsu_embeddings]

# 類似度の結果をデータフレームとして整形
similarity_df = pd.DataFrame(similarity_matrix, 
                            index=[f"{text} (ペット)" for text in pet_texts],
                            columns=[f"{text} (生物)" for text in seibutsu_texts])

# 表形式でマークダウンとして保存
similarity_df.to_markdown('similarity_table.md')
print("ペットと生物間の類似度テーブルを similarity_table.md に保存しました")

# ヒートマップの描画
plt.figure(figsize=(15, 8))
sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', 
            xticklabels=seibutsu_texts, 
            yticklabels=pet_texts)
plt.title('ペットと生物の埋め込みのコサイン類似度')
plt.xlabel('生物トークンの埋め込み')
plt.ylabel('ペットトークンの埋め込み')
plt.tight_layout()
plt.savefig('pet_seibutsu_similarity_heatmap.png')
plt.show()
