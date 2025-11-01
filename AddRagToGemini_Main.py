import os
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict

# LlamaIndex RAG Components
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.chat_engine import CondensePlusContextChatEngine

from google import genai
from google.genai import types
from dotenv import load_dotenv

import uvicorn

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


# モデル名
def initialize_rag_components() -> CondensePlusContextChatEngine:
    """
    RAGインデックスを構築し、チャットエンジンを返します。
    """
    
    # 1. LlamaIndexのグローバル設定
    Settings.llm = Gemini(
            model="gemini-2.5-flash",
            api_key=API_KEY  # 👈 settingsから直接渡す
        )
    Settings.embed_model = GeminiEmbedding(
            model_name="text-embedding-004",
            api_key=API_KEY  # 👈 settingsから直接渡す
        )

    # 2. 知識ベースの構築
    try:
        # 'docs'フォルダ内の全ファイルを読み込み
        documents = SimpleDirectoryReader("./docs").load_data()
    except Exception as e:
        print(f"警告: 'docs'フォルダの読み込みに失敗しました。{e}")
        documents = []

    if not documents:
        print("RAGインデックスの作成をスキップします。")
        # ドキュメントがない場合は、空のインデックスを作成
        index = VectorStoreIndex([])
    else:
        # ベクトルDBを構築（埋め込み生成と保存）
        index = VectorStoreIndex.from_documents(documents)
        print(f"RAGインデックス構築完了。ドキュメント数: {len(documents)}")

    # 3. RAGチャットエンジンの作成
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=index.as_retriever(),
        llm=Settings.llm,
        system_prompt="あなたは提供された知識ベースに基づいてのみ回答するプロフェッショナルなアシスタントです。知識ベースに情報がない場合は、申し訳ありませんが、その情報はありませんと伝えてください。",
        # 履歴を管理するためのストレージを設定することもできますが、今回はメモリ内で管理します。
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