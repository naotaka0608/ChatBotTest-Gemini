import os
import sys
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Any

# LlamaIndex RAG Components
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.chat_engine import CondensePlusContextChatEngine

from google import genai
from google.genai import types

import uvicorn


# --- 1. 設定の読み込みと環境チェック ---
class Settings(BaseSettings):
    """pydantic-settings が .env ファイルを自動で読み込みます。"""
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra='ignore'
    )
    GEMINI_API_KEY: str

settings = Settings()



if not settings.GEMINI_API_KEY:
    print("FATAL: 'GEMINI_API_KEY' が設定されていません。'.env' ファイルを確認してください。")
    sys.exit(1)
    
    
# --- 2. RAG エンジンの初期化 ---
def initialize_rag_components() -> CondensePlusContextChatEngine:
    """RAGインデックスを構築し、チャットエンジンを返します。"""

    api_key = settings.GEMINI_API_KEY
    print(f"DEBUG: API Key Loaded (starts with: {api_key[:5]})")
    
    # 2.1. Gemini Clientの初期化と認証チェック
    try:
        # LLM Client/Embedding Model に APIキーを直接渡す
        llm_client = Gemini(model="gemini-2.5-flash", api_key=api_key)
        embed_model = GeminiEmbedding(model_name="text-embedding-004", api_key=api_key)
        
        print("INFO: Gemini LLM/Embedding Client Initialization Successful.")
        
    except Exception as e:
        print(f"FATAL ERROR: Gemini Clientの初期化に失敗しました。APIキーを確認してください: {e}")
        sys.exit(1) 

    # 2.2. LlamaIndexのグローバル設定
    # 🚨 ServiceContextの代わりにSettingsに直接設定します 🚨

    Settings.llm = llm_client
    Settings.embed_model = embed_model
    

    # 2.3. 知識ベースの構築
    try:
        documents = SimpleDirectoryReader("./docs").load_data()
    except Exception as e:
        print(f"WARNING: 'docs'フォルダの読み込みに失敗しました。{e}")
        documents = []

    if not documents:
        print("WARNING: RAGインデックスの作成をスキップします。純粋なGeminiチャットとして動作します。")
        # Settingsが有効なため、引数なしでインデックスを作成
        index = VectorStoreIndex([], embed_model=embed_model)
    else:
        index = VectorStoreIndex.from_documents(
            documents, 
            embed_model=embed_model # 👈 embed_model を直接渡す
        )
        print(f"INFO: RAGインデックス構築完了。ドキュメント数: {len(documents)}")

    # 2.4. RAGチャットエンジンの作成
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=index.as_retriever(),
        llm=llm_client, 
        system_prompt="あなたは提供された知識ベースに基づいてのみ回答するプロフェッショナルなアシスタントです。知識ベースに情報がない場合は、その旨を丁寧に伝えてください。",
    )

    return chat_engine




# 🚨 アプリケーション起動時にRAGコンポーネントを一度だけ初期化 🚨
try:
    RAG_CHAT_ENGINE = initialize_rag_components()
except Exception as e:
    raise RuntimeError(f"RAGエンジンの初期化中に致命的なエラーが発生しました: {e}")


# --- 3. FastAPI アプリケーションの定義 ---
app = FastAPI()
chat_engines: Dict[str, BaseChatEngine] = {}

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    message: str


# --- 4. ストリーミングジェネレータ (非同期エラー回避版) ---
async def generate_rag_stream(engine: BaseChatEngine, prompt: str):
    """LlamaIndexのストリーミング応答を処理する非同期ジェネレータ"""
    try:
        # 非同期ストリーミングメソッドを呼び出し、応答オブジェクトを取得
        response_stream = await engine.astream_chat(prompt) 

        # 応答オブジェクト内の通常のジェネレータ (.response_gen) を通常の for で処理
        # 非同期コンテキストを維持するため、ループ内で await asyncio.sleep(0) を実行
        for token in response_stream.response_gen:
            if token:
                yield token 
                await asyncio.sleep(0)

    except Exception as e:
        print(f"RAG/APIエラー: {e}")
        yield f"\n[ERROR] RAG/APIエラーが発生しました: {e}"


# --- 5. エンドポイント ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_id = request.user_id
    prompt = request.message
    
    if user_id not in chat_engines:
        # 新しいセッションを作成し、履歴を独立させる
        chat_engines[user_id] = RAG_CHAT_ENGINE
        print(f"新規RAGセッション開始: {user_id}")
    
    chat_engine = chat_engines[user_id]
    
    return StreamingResponse(
        generate_rag_stream(chat_engine, prompt),
        media_type="text/plain" 
    )