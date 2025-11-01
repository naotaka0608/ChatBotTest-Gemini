# 説明
チャットボットを作成  
pythonのフレームワークFastAPIを使用  
Gemini APIと連携して、質問に対する回答はGeminiしてくれる  


## 開発環境

- python 3.12.10  
- pip 25.3  
- Gemini 1.47.0

2025/11/1現在、python 3.14.0ではうまくいかない


## Google AI Studio で APIキーを取得
Google AI Studio で APIキーを取得するには  
- Google AI Studio に移動  
- サイドパネルから [ダッシュボード]  
- [プロジェクト] を選択  
- [プロジェクト] ページで、[プロジェクトをインポート] ボタンを選択  
- インポートする Google Cloud プロジェクトを検索して選択し、[インポート] ボタンを選択  
- プロジェクトをインポートしたら、ダッシュボード メニューから API キーページに移動し、インポートしたプロジェクトで API キーを作成  


## 下記pipコマンドでインストール

- 仮想環境作成
```bash
$ pip install pipenv
$ pipenv --python 3
$ pipenv install fastapi uvicorn google-genai pydantic load_dotenv
```

- 仮想環境作成

・pipfileに下記追加  
```bash
[scripts]  
start1 = "uvicorn GeminiOnly_Main:app --reload"  
start2 = "uvicorn AddRagToGemini_Main:app --reload"  
```

## .envファイル
.envファイルを作成して、GeminiのAPIキーを入力
```txt
GEMINI_API_KEY="ここに取得したAPIキーを貼り付け"
```

・サーバーの実行 
```bash
$ pipenv run start1
```



GeminiOnly_HP.htmlを直接開く




## RAG追加
```bash
pipenv install llama-index llama-index-llms-gemini llama-index-embeddings-gemini pypdf
```



## その他
いらなくなったpipenvの環境は下記フォルダ内の仮想環境フォルダを削除
```bash
%userprofile%\.virtualenvs
```
