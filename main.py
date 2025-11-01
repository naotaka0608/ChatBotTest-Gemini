import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv


# ----------------------------------------------------
# 1. .envファイルをロード
# ----------------------------------------------------
# これにより、.envファイル内のキーが os.environ に読み込まれます
load_dotenv()

# 🚨 注意: 本番環境では環境変数を使用してください
# os.environ["GEMINI_API_KEY"] が設定されている前提
# client = genai.Client()
# テストのため、直接キーを指定する（非推奨）
#API_KEY = "ここに取得したAPIキーを入力"  # <<< ここにご自身のAPIキーを貼り付けてください
API_KEY = os.getenv("GEMINI_API_KEY")

try:

    client = genai.Client(api_key=API_KEY) # 環境変数から読み込む
    
except Exception as e:
    raise RuntimeError(f"Gemini Clientの初期化に失敗: {e}")

app = FastAPI()

# ... 既存のMODEL_NAME, chat_sessions, ChatRequest の定義 ...

# 🚨 ここから CORS 設定を追加 🚨
origins = [
    # 開発環境でFastAPIとは異なるポートを使用している場合に必要
    "http://127.0.0.1:5500", # 例: VS CodeのLive Serverなど
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    # もし index.html をローカルファイルとして開いている場合は、'*' を使用する必要があるかもしれません。
    # しかし、セキュリティ上は具体的なオリジンを指定するのがベストです。
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # テストのためにすべてのオリジンを許可（本番では必ず具体的なオリジンを指定）
    allow_credentials=True,
    allow_methods=["*"], # OPTIONS, POST などを許可
    allow_headers=["*"], # すべてのヘッダーを許可
)
# 🚨 CORS 設定 ここまで 🚨

MODEL_NAME = 'gemini-2.5-flash'

# 会話履歴を保持するためのダミーのストレージ（本番ではデータベースやセッション管理が必要です）
# ユーザーID: [chat_session] の形式で保存することを想定
chat_sessions = {}

# リクエストボディの定義
class ChatRequest(BaseModel):
    user_id: str # ユーザーを特定するためのID
    message: str

# 非同期ジェネレータ関数: ストリーミング応答のために必要
async def generate_response_stream(chat_session, prompt: str):
    """
    Gemini APIからストリーミング応答を受け取り、クライアントに逐次送信するジェネレータ
    """
    try:
        # Gemini APIのストリーミング呼び出し
        response_stream = chat_session.send_message_stream(prompt)
        
        for chunk in response_stream:
            # chunk.text は通常、応答の一部（チャンク）を含みます
            if chunk.text:
                yield chunk.text
    except Exception as e:
        print(f"Gemini APIエラー: {e}")
        # クライアント側でエラーを処理できるよう、エラーメッセージを送信
        yield f"\n[ERROR] 対話中に問題が発生しました: {e}"


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    prompt = request.message
    
    # ユーザーのチャットセッションを取得または新規作成
    if user_id not in chat_sessions:
        # 新しいチャットセッションを作成 (システム命令を設定)
        chat_sessions[user_id] = client.chats.create(
            model=MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction="あなたは役立つAIアシスタントです。質問には簡潔に答えてください。"
            )
        )
        print(f"新規セッション開始: {user_id}")

    chat_session = chat_sessions[user_id]
    
    # ストリーミング応答を返す
    return StreamingResponse(
        generate_response_stream(chat_session, prompt),
        media_type="text/plain" # または "text/event-stream"
    )
