import os
# RAG Service のインポート修正例
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core import Settings
# from llama_index.core.embeddings.types import BaseEmbedding # 👈 古いパス
from llama_index.core.embeddings import BaseEmbedding # 👈 新しいパス（ただし、通常は不要なことが多い）
from dotenv import load_dotenv

load_dotenv()

# 環境変数からAPIキーを設定 (LlamaIndexが使用)
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY") 

# モデル名
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-004"


def initialize_rag_components() -> CondensePlusContextChatEngine:
    """
    RAGインデックスを構築し、チャットエンジンを返します。
    """
    
    # 1. LlamaIndexのグローバル設定
    Settings.llm = Gemini(model=LLM_MODEL)
    Settings.embed_model = GeminiEmbedding(model_name=EMBEDDING_MODEL)

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