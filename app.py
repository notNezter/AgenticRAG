# app.py
# Import the AutoGen utilities
import autogen
from autogen import Agent, AssistantAgent, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

# Import custom database methods
from database import init_db, insert_data, insert_processed_file, ProcessedData, query_data, query_processed_files, format_for_agent

# Import the FastAPI utilities
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

# Import the local vectorizing model
from sentence_transformers import SentenceTransformer

from typing import Annotated

# Import the system utilities
import asyncio
import logging
import json
import os
import PyPDF2
import uuid

"""# Import the uvicorn server
import uvicorn
"""
# Initialize logging
logging.basicConfig(level=logging.INFO)

# Inititialize the FastAPI application
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Define the path to the docs directory
docs_path = "docs"

# Ensure the docs directory exists
if not os.path.exists(docs_path):
    os.makedirs(docs_path)

# Initialize SentenceTransformer for using a local model
vector_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load OpenAI API Key from Environment Variables
api_key = os.environ.get('OPENAI_API_KEY')

# Get the list of models from OAI_CONFIG_LIST
config_list = config_list_from_json("OAI_CONFIG_LIST")

# Custom token counter method
def custom_token_count_function(text, model):
    return len(text.split())

# Check if the last word in content is TERMINATE
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

# Settings for the LLM(s). It's a list of dictionaries, so you can actually configure multiple models
llm_config = {
    "config_list": config_list,
    "api_key": api_key,
    "temperature": 0.0,
    "price": [0, 0], # if you use OpenAI or another service, comment this out
    "cache_seed": 772293, # use a randomizer if you want different answers when using the same prompt - can also configure the interface to pass this as a user variable. AutoGen Studio has a nice cache factory feature that I like to use.
    "timeout": 300,
}

"""
Initialize agents
For the system messages, you can create a ReAct prompt, then end the entire prompt with, "Reply `TERMINATE` in the end when everything is done.",
so that the termination_msg method auto fills is_termination_msg and automatically terminates each response when complete.
"""
assistant = AssistantAgent(
    name="assistant",
    is_termination_msg=termination_msg,
    system_message="You are a helpful assistant. Be thoughtful in decomposing the request. Make a plan of action. Keep your responses short and to the point. When providing code, put the code into a block, indicating what type of code it is (e.g., ```Python, ###C or ###C++, etc). If you do not know something, state `MORE CONTEXT REQUIRED`. When everything is completed, reply with `TERMINATE`.",
    llm_config=llm_config,
)

arbiter = AssistantAgent(
    name="arbiter",
    is_termination_msg=termination_msg,
    system_message="You are a critic that reviews the responses from other agents. You ensure that whatever the other agents produce is not a hallucination or made up fact. If something is found to not be ground in truth (e.g., no reliable sources can be provided), point it out to the other agents so they can correct accordingly. Keep your feedback short and to the point. If there is nothing to fix, then simply reply with `TERMINATE.` Otherwise, when everything is completed to your satisfaction, reply with `TERMINATE`.",
    llm_config=llm_config,
)

dbassistant = RetrieveAssistantAgent(
    name="dbassistant",
    is_termination_msg=termination_msg,
    system_message="You are an expert database administrator.",
    llm_config=llm_config,
)

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
    default_auto_reply="Reply `TERMINATE` in the end when everything is done.",
    retrieve_config={
        "task": "qa", # options are code, qa, or default
        "docs_path": [
            os.path.join(os.path.abspath(""),
            "docs"),
        ],
        "custom_text_types": ["txt"],
        "chunk_token_size": 2000,
        "model": vector_model, # use "vector_db": "chroma",
        "get_or_create": True,
        "overwrite": True,
        "custom_token_count_function": custom_token_count_function, # comment this out if you use OpenAI
        "must_break_at_empty_line": False
    },
    code_execution_config=False,
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER", # options are ALWAYS, TERMINATE, NEVER
    code_execution_config=False,
    default_auto_reply="Reply `TERMINATE` in the end when everything is done.",
)

# Establish list of RAG agents
rag_agents = [ragproxyagent, dbassistant]

# Establish the list of non RAG agents
non_rag_agents = [user_proxy, assistant]

# Establish a default Problem
PROBLEM = "Summarize this document"

# Reset agents between interactions
def _reset_agents():
    assistant.reset()
    arbiter.reset()
    dbassistant.reset()
    ragproxyagent.reset()
    user_proxy.reset()

# Create a group chat with RAG
def rag_chat(query: str = None, file_content: str = None) -> str:
    _reset_agents()
    groupchat = GroupChat(
        agents = rag_agents,
        messages = [],
        max_round = 10,
        speaker_selection_method= "round_robin",
        allow_repeat_speaker=False,
    )
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )

    message = ""
    if file_content and query:
        message = f"Here is the file content: {file_content}. The user query is: {query}"
    elif file_content:
        message = f"Here is the file content: {file_content}"
    elif query:
        message = query
    else:
        message = PROBLEM  # Fallback to the default problem

    # Start the conversation with ragproxyagent as this is the retrieval user proxy
    ragproxyagent.initiate_chat(
        manager,
        message=ragproxyagent.message_generator,
        problem=message,
        n_results = 3,
    )

    for msg in messages:
        if msg.get("name") == "assistant" and "TERMINATE" not in msg["content"]:
            compiled_responses.append(msg["content"])
        elif msg.get("name") != "user_proxy" and "TERMINATE" not in msg["content"]:
            compiled_responses.append(f"{msg['name']}: {msg['content']}")

    final_response = "\n\n".join(compiled_responses)

    if "TERMINATE" in final_response.upper():
        _reset_agents()
        final_response = final_response.replace("TERMINATE", "").strip()

    return final_response

def norag_chat(query: str) -> str:
    _reset_agents()
    groupchat = GroupChat(
        agents = non_rag_agents,
        messages = [],
        max_round = 10,
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
    )
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )

    # Start the conversation with the user_proxy as this is the non RAG user proxy
    user_proxy.initiate_chat(
        manager,
        message=query,
        silent=False,
        summary_prompt="Summarize the takeaway from the responses. Do not add any introductory phrases. If the intended request is NOT properly addressed, point it out. Strip out any parts of the response that do not make sense (aka hallucinations).",
        summary_method="reflection_with_llm",
    )

    messages = groupchat.messages
    compiled_responses = []

    for msg in messages:
        if msg.get("name") == "assistant":
            compiled_responses.append(msg["content"].replace("TERMINATE", "").strip())
        elif msg.get("name") != "user_proxy":
            compiled_responses.append(f"{msg['name']}: {msg['content']}")

    return "\n\n".join(compiled_responses)

"""
The chat will be initiated by the user_proxy as that is the human stand in.
The RAG part of the app will be wrapped in a function to be
called by the other agents.
"""
def call_rag_chat(file_content=None, query=None):
    _reset_agents()

    def retrieve_content(
        message: Annotated[
            str,
            "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
        ],
        n_results: Annotated[int, "number of results" ] = 3,
    ) -> str:
        ragproxyagent.n_results = n_results # Set the number of results to retrieve
        _context = {"problem": message, "n_results": n_results}
        ret_msg = ragproxyagent.message_generator(ragproxyagent, None, _context)
        return ret_msg or message

    ragproxyagent.human_input_mode = "NEVER" # Disable human input as it should only retrieve content

    # Register functions for retrieval and execution
    for caller in [assistant, dbassistant]:
        d_retrieve_content = caller.register_for_llm(
            description = "retrieve content for code generation and question answering.",
            api_style="function"
        )(retrieve_content)

    for executor in [user_proxy,ragproxyagent]:
        d_retrieve_content = executor.register_for_execution()(d_retrieve_content)

    groupchat = GroupChat(
        agents = non_rag_agents,
        messages=[],
        max_round=10,
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False,
    )
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )

    # Combine the file content and query
    message = ""
    if file_content and query:
        message = f"Here is the file content: {file_content}. The user query is: {query}"
    elif file_content:
        message = f"Here is the file content: {file_content}"
    elif query:
        message = query
    else:
        message = PROBLEM # Fallback to the default problem

    # Start chatting with the user_proxy as this is the user proxy agent
    user_proxy.initiate_chat(
        manager,
        message=message,
        silent=False,
        summary_prompt="Summarize the takeaway from the responses. Do not add any introductory phrases. If the intended request is NOT properly addressed, point it out. Strip out any parts of the response that do not make sense (aka hallucinations).",
        summary_method="reflection_with_llm",
    )

    # Extract the conversation messages from the group chat
    messages = groupchat.messages
    compiled_responses = []

    # Loop through messages and compile meaningful responses
    for msg in messages:
        if msg.get("name") == "assistant" and "TERMINATE" not in msg["content"]:
            compiled_responses.append(msg["content"])
        elif msg.get("name") != "user_proxy" and "TERMINATE" not in msg["content"]:
            compiled_responses.append(f"{msg['name']}: {msg['content']}")

    # Join all assistant's responses into one final result
    final_response = "\n\n".join(compiled_responses)

    # Check if termination is needed
    if "TERMINATE" in final_response.upper():
        # Manually trigger termination by resetting agents and stopping the chat loop
        _reset_agents()  # Ensure the agents are reset after the task is done
        final_response = final_response.replace("TERMINATE", "").strip()  # Clean up the response

    return final_response

# Register the query_database function with dbassistant
@dbassistant.register_for_llm(name="query_database", description="Tool for querying this application database")
def query_database() -> list[ProcessedData]:
    data = query_data()
    formatted_data = format_for_agent(data)
    return formatted_data

# Initialize database
init_db()

"""
@app.get("/")
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
"""

# Serve the SPA index.html for all routes
@app.get("/{path_name:path}")
async def serve_spa(path_name: str):
    return FileResponse('templates/index.html')
    
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(None), query: str = Form(...)):
    try:
        # Check if query is empty or missing
        if not query:
            return JSONResponse(content={"error": "Query is missing"}, status_code=400)

        if file is None:
            return JSONResponse(content={"error": "No file provided"}, status_code=400)

        # Extract the file content
        file_path = os.path.join(docs_path, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process the uploaded file
        all_content = ""
        if file.filename.endswith(".pdf"):
            with open(file_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(reader.pages)
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    all_content += page.extract_text()
        elif file.filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as txt_file:
                all_content += txt_file.read()
        else:
            return JSONResponse(content={"error": "Unsupported file format"}, status_code=400)

        # Debug: Print the extracted content
        print("Extracted content:", all_content)

        """# Determine if RAG is needed or not
        if needs_retrieval(query):
            result = call_rag_chat(query=query, file_content=all_content)
        else:
            result = norag_chat(query)"""

        # Call RAG chat with both the query and the file content
        result = call_rag_chat(query=query, file_content=all_content)
        print("Chat Result:", result)

        # Insert data into the main database
        insert_data(file.filename, file_path, all_content)

        return JSONResponse(content={"summary": result})
    except Exception as e:
        print(f"Error in /uploadfile/: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)
        
@app.post("/chat/")
async def chat_with_agent(request: Request):
    try:
        data = await request.json()
        query = data.get("query", "").strip()  # Use 'query' for consistency

        if not query:
            return JSONResponse(content={"error": "Query is missing"}, status_code=400)

        # Pass the query to norag_chat
        result = call_rag_chat(query)
        print("Chat Result:", result)

        return JSONResponse(content={"response": result})
    except Exception as e:
        print(f"Error in /chat/: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)

if __name__ == "__main__":
    import uvicorn

    # To run on Windows: uvicorn app:app --host 127.0.0.1 --port 8005 --reload
    try:
        uvicorn.run(app, host="127.0.0.1", port=8005, reload=True)
    except Exception as e:
        logging.error(f"Failed to start the server: {e}")
