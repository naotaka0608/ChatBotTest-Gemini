# 説明
チャットボットを作成  
pythonのフレームワークFastAPIを使用  
Gemini APIと連携して、質問に対する回答はGeminiしてくれる  


## 開発環境
python 3.10.6  
pip 25.3  
Gemini 1.47.0

## Google AI Studio で APIキーを取得
Google AI Studio で APIキーを取得するには  
・Google AI Studio に移動  
・サイドパネルから [ダッシュボード]  
・[プロジェクト] を選択  
・[プロジェクト] ページで、[プロジェクトをインポート] ボタンを選択  
・インポートする Google Cloud プロジェクトを検索して選択し、[インポート] ボタンを選択  
・プロジェクトをインポートしたら、ダッシュボード メニューから API キーページに移動し、インポートしたプロジェクトで API キーを作成  


## 下記pipコマンドでインストール
```bash
$ pip install fastapi uvicorn google-genai pydantic
```
又は
```bash
$ pip install -r requirements.txt
```


## 実行
サーバーの実行  
```bash
$ uvicorn main:app --reload
```

index.htmlを直接開く

