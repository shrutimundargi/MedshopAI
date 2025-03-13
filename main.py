import os
import logging
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import uvicorn
from operator import itemgetter
from pydantic_settings import BaseSettings
from langchain_core.runnables import RunnableLambda

# --------------------------
# Configuration Management
# --------------------------
class Settings(BaseSettings):
    database_url: str = Field(..., env="DATABASE_URL")
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    app_env: str = Field("development", env="APP_ENV")
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# --------------------------
# Logging Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------
# Application Setup
# --------------------------
app = FastAPI(
    title="Medical CRM NLQ API",
    description="Natural Language Query Interface for Medical CRM",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --------------------------
# Database Connection
# --------------------------
def get_db_connection() -> SQLDatabase:
    try:
        db = SQLDatabase.from_uri(
            settings.database_url,  
            sample_rows_in_table_info=3
        )
        logger.info("Database connection established")
        return db
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection error"
        )

# --------------------------
# AI Service Layer
# -------------------------- 

class AIService:
    def __init__(self, db: SQLDatabase):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=settings.openai_api_key,
            max_retries=3
        )
        self.db = db
        self.chain = self.run_chain_llm()
        
    def run_chain_llm(self):
        write_query = create_sql_query_chain(self.llm, self.db)
        execute_query = QuerySQLDataBaseTool(db=self.db)
        answer_prompt = PromptTemplate.from_template(
            """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

            Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            Answer: """
            )

        return (
            RunnablePassthrough.assign(query=write_query).assign(
                result=itemgetter("query") | execute_query
            )
            | answer_prompt
            | self.llm
            | StrOutputParser()
        )



# --------------------------
# Dependency Injection
# --------------------------
def get_ai_service(db: SQLDatabase = Depends(get_db_connection)) -> AIService:
    return AIService(db)

# --------------------------
# Request/Response Models
# --------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., example="Give me list all the medicines in stock at store 3")

class QueryResponse(BaseModel):
    result: str

# --------------------------
# API Endpoints
# --------------------------
@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    ai_service: AIService = Depends(get_ai_service)
):
    """Process natural language query about medical inventory"""
    try:
        logger.info(f"Processing query: {request.question}")
        result = ai_service.chain.invoke({
            "question": f"{request.question}"
        })
        return {
            "result": result
        }
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query"
        )

@app.get("/health")
async def health_check():
    """Service health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.app_env,
        "database_connected": bool(settings.database_url)
    }

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=settings.app_env == "development"
    )
