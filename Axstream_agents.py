import streamlit as st



# Create a function for each page
def home():
    import os
    from langchain.text_splitter import CharacterTextSplitter
    #from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.document_loaders import PyPDFLoader
    from langchain.document_loaders import TextLoader
    from langchain.vectorstores import Pinecone
    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.chains.question_answering import load_qa_chain
    from langchain.chains import ConversationChain
    from langchain.chains.conversation.memory import ConversationBufferMemory
    import pinecone
    #openai_api_key = "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"

    os.environ["PINECONE_API_KEY"] = pineconekey
    os.environ["OPENAI_API_KEY"] = openai_api_key



    st.title("🦜AXstreaM🤖 AGENTS")
    # Add content specific to the home page
    #import streamlit as st
    import requests

    API_URLS = [
        "https://flowise-production-a606.up.railway.app/api/v1/prediction/8f17b231-6b0c-4ab6-929d-214d368e111e",  # DOCgpt
        "https://flowise-production-a606.up.railway.app/api/v1/prediction/6a421494-72c9-42f1-9520-a84591bbdc54",  # GoogleGPT
        "https://flowise-production-a606.up.railway.app/api/v1/prediction/fafd1f25-a3a9-4c63-ad61-9cd8aa9100ad", # English to malay Translator
        "https://flowise-production-a606.up.railway.app/api/v1/prediction/09095057-e5c7-4a41-ad72-0c5f8ec77c54",  # baby agi
        "https://flowise-production-a606.up.railway.app/api/v1/prediction/8f17b231-6b0c-4ab6-929d-214d368e111e" #finetune answer from doc - web - openai

    ]

    # Initialize the selected API index
    selected_api_index = 0

    def query(payload):
        try:
            response = requests.post(API_URLS[selected_api_index], json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as err:
            st.error("Error occurred:", err)
            return None
        
       
    def query1(payload):
        try:
            response = requests.post(API_URLS[1], json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as err:
            st.error("Error occurred:", err)
            return None
         
    









    #st.markdown(custom_style, unsafe_allow_html=True)

    #st.title("ASRIX META_Chatbot 🤖")

    # Sidebar
    #st.sidebar.title("Select Below 🦜CHAT-BOTS")

    selected_api_index = st.radio(
        "SELECT 🦜Agents BELOW ",
        list(range(len(API_URLS))),
        format_func=lambda i: "QA based on document (pdf/url)" if i == 0 else "QA on current info using web search" if i == 1 else "Translate English to Malay" if i == 2 else "BABY AGI" if i==3 else "Finetune answers Doc-Web-Openai",
    )


    #st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # List to hold the conversation history
    conversation = []
    chatlist =[]

    if selected_api_index== 0 :
        # Chat container

        # PDF UPLOADER



        docs_chunks = []

        # openai_api_key = os.environ.get('API_KEY')
        #openai_api_key = "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"
        #pineconekey = "f4e3f5b8-fc9a-4d6d-be18-ba5f200e0e52"
        #pineconeEnv = "us-west1-gcp-free"


        # Initialize Pinecone
        pinecone.init(api_key=pineconekey, environment=pineconeEnv)
        #index_name2 = "babyagi"

        # embeddings

        embeddings = OpenAIEmbeddings()

        # image = Image.open('ai.png')
        # st.image(image, caption='AI', width=200)



        


        def process_file(uploaded_file):

            bytes_data = uploaded_file.read()
            st.write("filename:", uploaded_file.name)
            _, file_extension = os.path.splitext(uploaded_file.name)

            # write the uploaded file to disk
            with open(uploaded_file.name, 'wb') as f:
                f.write(bytes_data)

            documents = None
            if file_extension.lower() == '.pdf':
                # Load the PDF file with PyPDF Loader
                loader = PyPDFLoader(uploaded_file.name)
                documents = loader.load()

            elif file_extension.lower() == '.txt':
                # Load the text file with TextLoader
                loader = TextLoader(uploaded_file.name, encoding='utf8')
                documents = loader.load()

            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            return documents

        def split_docs(documents, chunk_size=1000, chunk_overlap=20):
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            docs = text_splitter.split_documents(documents)
            return docs

        uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True)
        if st.button('upload'):

            if len(uploaded_files) > 0:
                for uploaded_file in uploaded_files:
                    documents = process_file(uploaded_file)
                    docs_chunks.extend(split_docs(documents))



        index1 = Pinecone.from_documents(docs_chunks, embeddings, index_name=index_name2, overwrite=True)

    #selected_api_index == 4

    # User input
    user_input = st.text_input("User:", key="user_input")
    submit_button = st.button("ASK", key="submit_button")





    #if selected_api_index==4:







    if submit_button and user_input:
        response = query({"history": conversation, "question": user_input})
        response1 = query1({"history": conversation, "question": user_input})
        #response1 = query1({"history": conversation, "question": user_input})
        # Add user input to conversation history
        conversation.append({"role": "user", "content": user_input})
        # Query the selected API

        if response is not None :
            # Add bot response to conversation history
            conversation.append({"role": "bot", "content": response})
        else:
            conversation.append(
                {
                    "role": "bot",
                    "content": "Sorry, I am unable to process your request at the moment.",
                }
            )
         


        
        #st.write("Answer:-", response)
        
        if selected_api_index !=4 :
            st.write("Answer:-", response)
            
            
            
        if selected_api_index == 4:
            
            if response or response1 is not None:
                # Add bot response to conversation history
                chatlist.append({"google result": response1, "document result": response})
            else:
                chatlist.append(
                    {
                        "result": "bot",
                        "content": "Sorry, I am unable to process your request at the moment.",
                    }
                )


            model_name = "gpt-3.5-turbo"
            llm = ChatOpenAI(temperature=0.2, model_name=model_name)
            conversation1 = ConversationChain(
                llm=llm,
                verbose=True,
                memory=ConversationBufferMemory()
            )
            answer = conversation1.predict(
                input=f"understand the question{user_input} just answer it from this content given{chatlist} ")
            st.write("answer3", answer)
            




        





    # Display the conversation history

   

def about():
    st.title("🦜AXstream-BABY AGI 👼🤖")
    # Add content specific to the about page
    import os
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["SERPAPI_API_KEY"] = serp_api



    #import streamlit as st
    from collections import deque
    from typing import Dict, List, Optional, Any

    from langchain import LLMChain, OpenAI, PromptTemplate
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.llms import BaseLLM
    from langchain.vectorstores.base import VectorStore
    from pydantic import BaseModel, Field
    from langchain.chains.base import Chain
    from langchain.vectorstores import FAISS
    from langchain.docstore import InMemoryDocstore

    st.title('Autonomous AI agent - 🔗(internet)')

    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    import faiss

    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    class TaskCreationChain(LLMChain):
        """Chain to generates tasks."""

        @classmethod
        def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
            """Get the response parser."""
            task_creation_template = (
                "You are an task creation AI that uses the result of an execution agent"
                " to create new tasks with the following objective: {objective},"
                " The last completed task has the result: {result}."
                " This result was based on this task description: {task_description}."
                " These are incomplete tasks: {incomplete_tasks}."
                " Based on the result, create new tasks to be completed"
                " by the AI system that do not overlap with incomplete tasks."
                " Return the tasks as an array."
            )
            prompt = PromptTemplate(
                template=task_creation_template,
                input_variables=[
                    "result",
                    "task_description",
                    "incomplete_tasks",
                    "objective",
                ],
            )
            return cls(prompt=prompt, llm=llm, verbose=verbose)

    class TaskPrioritizationChain(LLMChain):
        """Chain to prioritize tasks."""

        @classmethod
        def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
            """Get the response parser."""
            task_prioritization_template = (
                "You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing"
                " the following tasks: {task_names}."
                " Consider the ultimate objective of your team: {objective}."
                " Do not remove any tasks. Return the result as a numbered list, like:"
                " #. First task"
                " #. Second task"
                " Start the task list with number {next_task_id}."
            )
            prompt = PromptTemplate(
                template=task_prioritization_template,
                input_variables=["task_names", "next_task_id", "objective"],
            )
            return cls(prompt=prompt, llm=llm, verbose=verbose)

    from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
    from langchain import OpenAI, SerpAPIWrapper, LLMChain

    todo_prompt = PromptTemplate.from_template(
        "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
    )
    todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events",
        ),
        Tool(
            name="TODO",
            func=todo_chain.run,
            description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
        ),
    ]

    prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
    suffix = """Question: {task}
    {agent_scratchpad}"""
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["objective", "task", "context", "agent_scratchpad"],
    )

    def get_next_task(
            task_creation_chain: LLMChain,
            result: Dict,
            task_description: str,
            task_list: List[str],
            objective: str,
    ) -> List[Dict]:
        """Get the next task."""
        incomplete_tasks = ", ".join(task_list)
        response = task_creation_chain.run(
            result=result,
            task_description=task_description,
            incomplete_tasks=incomplete_tasks,
            objective=objective,
        )
        new_tasks = response.split("\n")
        return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]

    def prioritize_tasks(
            task_prioritization_chain: LLMChain,
            this_task_id: int,
            task_list: List[Dict],
            objective: str,
    ) -> List[Dict]:
        """Prioritize tasks."""
        task_names = [t["task_name"] for t in task_list]
        next_task_id = int(this_task_id) + 1
        response = task_prioritization_chain.run(
            task_names=task_names, next_task_id=next_task_id, objective=objective
        )
        new_tasks = response.split("\n")
        prioritized_task_list = []
        for task_string in new_tasks:
            if not task_string.strip():
                continue
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = task_parts[0].strip()
                task_name = task_parts[1].strip()
                prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
        return prioritized_task_list

    def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
        """Get the top k tasks based on the query."""
        results = vectorstore.similarity_search_with_score(query, k=k)
        if not results:
            return []
        sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
        return [str(item.metadata["task"]) for item in sorted_results]

    def execute_task(
            vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5
    ) -> str:
        """Execute a task."""
        context = _get_top_tasks(vectorstore, query=objective, k=k)
        return execution_chain.run(objective=objective, context=context, task=task)

    class BabyAGI(Chain, BaseModel):
        """Controller model for the BabyAGI agent."""

        task_list: deque = Field(default_factory=deque)
        task_creation_chain: TaskCreationChain = Field(...)
        task_prioritization_chain: TaskPrioritizationChain = Field(...)
        execution_chain: AgentExecutor = Field(...)
        task_id_counter: int = Field(1)
        vectorstore: VectorStore = Field(init=False)
        max_iterations: Optional[int] = None

        class Config:
            """Configuration for this pydantic object."""

            arbitrary_types_allowed = True

        def add_task(self, task: Dict):
            self.task_list.append(task)

        def print_task_list(self):
            print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
            for t in self.task_list:
                print(str(t["task_id"]) + ": " + t["task_name"])

        def print_next_task(self, task: Dict):
            print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
            print(str(task["task_id"]) + ": " + task["task_name"])
            return (str(task["task_id"]) + ": " + task["task_name"])

        def print_task_result(self, result: str):
            print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
            return (result)

        @property
        def input_keys(self) -> List[str]:
            return ["objective"]

        @property
        def output_keys(self) -> List[str]:
            return []

        @st.cache_data
        def _call(_self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            result_list = []
            """Run the agent."""
            objective = inputs["objective"]
            first_task = inputs.get("first_task", "Make a todo list")
            _self.add_task({"task_id": 1, "task_name": first_task})
            num_iters = 0
            while True:
                if _self.task_list:
                    _self.print_task_list()

                    # Step 1: Pull the first task
                    task = _self.task_list.popleft()
                    _self.print_next_task(task)
                    st.write('**Task:** \n')
                    st.write(_self.print_next_task(task))

                    # Step 2: Execute the task
                    result = execute_task(
                        _self.vectorstore, _self.execution_chain, objective, task["task_name"]
                    )
                    this_task_id = int(task["task_id"])
                    _self.print_task_result(result)
                    st.write('**Result:** \n')
                    st.write(_self.print_task_result(result))
                    result_list.append(result)

                    # Step 3: Store the result in Pinecone
                    result_id = f"result_{task['task_id']}"
                    _self.vectorstore.add_texts(
                        texts=[result],
                        metadatas=[{"task": task["task_name"]}],
                        ids=[result_id],
                    )

                    # Step 4: Create new tasks and reprioritize task list
                    new_tasks = get_next_task(
                        _self.task_creation_chain,
                        result,
                        task["task_name"],
                        [t["task_name"] for t in _self.task_list],
                        objective,
                    )
                    for new_task in new_tasks:
                        _self.task_id_counter += 1
                        new_task.update({"task_id": _self.task_id_counter})
                        _self.add_task(new_task)
                    _self.task_list = deque(
                        prioritize_tasks(
                            _self.task_prioritization_chain,
                            this_task_id,
                            list(_self.task_list),
                            objective,
                        )
                    )
                num_iters += 1
                if _self.max_iterations is not None and num_iters == _self.max_iterations:
                    print(
                        "\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m"
                    )
                    st.success('Task Completed!', icon="✅")
                    break

            # Create a temporary file to hold the text
            with open('output.txt', 'w') as f:
                for item in result_list:
                    f.write(item)
                    f.write("\n\n")

            return {}

        @classmethod
        def from_llm(
                cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = False, **kwargs
        ) -> "BabyAGI":
            """Initialize the BabyAGI Controller."""
            task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)
            task_prioritization_chain = TaskPrioritizationChain.from_llm(
                llm, verbose=verbose
            )
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            tool_names = [tool.name for tool in tools]
            agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True
            )
            return cls(
                task_creation_chain=task_creation_chain,
                task_prioritization_chain=task_prioritization_chain,
                execution_chain=agent_executor,
                vectorstore=vectorstore,
                **kwargs,
            )

    def get_text():
        input_text = st.text_input("Give the goal and subgoals for AI agent ", key="input")
        return input_text

    user_input = get_text()
    num_loops = st.text_input("Enter the number of iterations", value="3")

    def main():


        OBJECTIVE = user_input
        llm = OpenAI(temperature=0)
        # Logging of LLMChains
        verbose = False
        # If None, will keep on going forever. Customize the number of loops you want it to go through.

        max_iterations = int(num_loops) if num_loops.isdigit() else 3
        #max_iterations: Optional[int] = 2
        baby_agi = BabyAGI.from_llm(
            llm=llm, vectorstore=vectorstore, verbose=verbose, max_iterations=max_iterations
        )

        if user_input:
            baby_agi({"objective": OBJECTIVE})

            # Download the file using Streamlit's download_button() function
            st.download_button(
                label='Download Results',
                data=open('output.txt', 'rb').read(),
                file_name='output.txt',
                mime='text/plain'
            )

    if __name__ == "__main__":
        submit_button = st.button("ASK", key="submit_button")
        if submit_button:
            main()

    # baby_agi({"objective": OBJECTIVE})


def contact():
    #st.title("Contact Page")

    # Add content specific to the contact page
    import os
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.document_loaders import PyPDFLoader
    from langchain.document_loaders import TextLoader
    from langchain.vectorstores import Pinecone
    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.chains.question_answering import load_qa_chain
    import pinecone
    from langchain.vectorstores import FAISS

    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

    #openai_api_key = os.environ.get('API_KEY')
    #openai_api_key= "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"
    #pineconekey = "f4e3f5b8-fc9a-4d6d-be18-ba5f200e0e52"
    #pineconeEnv = "us-west1-gcp-free"
    os.environ["PINECONE_API_KEY"] = pineconekey
    os.environ["OPENAI_API_KEY"] = openai_api_key

    embeddings = OpenAIEmbeddings()

    # Initialize Pinecone
    pinecone.init(api_key=pineconekey, environment=pineconeEnv)
    index_name2 = "axstream"

    st.title('🦜XstreaM  CHAT💬--DOC 📄')


    uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True)


    def process_file(uploaded_file):
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        _, file_extension = os.path.splitext(uploaded_file.name)

        # write the uploaded file to disk
        with open(uploaded_file.name, 'wb') as f:
            f.write(bytes_data)

        documents = None
        if file_extension.lower() == '.pdf':
            # Load the PDF file with PyPDF Loader
            loader = PyPDFLoader(uploaded_file.name)
            documents = loader.load()

        elif file_extension.lower() == '.txt':
            # Load the text file with TextLoader
            loader = TextLoader(uploaded_file.name, encoding='utf8')
            documents = loader.load()

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return documents



    def split_docs(documents, chunk_size=1000, chunk_overlap=20) :
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        docs = text_splitter.split_documents(documents)
        return docs

    docs_chunks = []

    if st.button('upload'):

        if len(uploaded_files) > 0:
            for uploaded_file in uploaded_files:
                documents = process_file(uploaded_file)
                docs_chunks.extend(split_docs(documents))

    index1 = Pinecone.from_documents(docs_chunks, embeddings, index_name=index_name2)

   # Prompt Template

    prompt = st.text_input("Ask a question about the PDF content:")

    script_template = PromptTemplate(
        input_variables=['question', 'context'],
        template='Write an answer for this: {question}\n{context}\nAnswer: '
    )


    # Llms
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0.2, model_name=model_name)
    script_chain = LLMChain(llm = llm, prompt = script_template, verbose = True, output_key = 'question')

    #if selected_api_index == 0:
    #uploaded_files = st.file_uploader("Choose PDF file", accept_multiple_files=True)


    # find similar documnets
    def get_similiar_docs(query, k=2, score = False) :

        if score :
            similar_docs = index1.similarity_search_with_score(query, k=k)
        else :
            similar_docs = index1.similarity_search(query, k=k)
        return similar_docs


    #similar_docs = get_similiar_docs(prompt)


    # question answering chain
    #similar_docs = get_similiar_docs(prompt)
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)



    #answer1 = chain.run(input_documents=similar_docs, question=f"understand the {similar_docs} and try to give the meaningful answer and give the step by step answer for {prompt} ")


    def get_answer(query):
        # Get similar documents
        similar_docs = get_similiar_docs(query)
        # Generate answer using script_chain
        answer = chain.run(input_documents=similar_docs, question=query)
        return answer



    
    if st.button('Ask'):
        try:
            st.spinner("Processing")
            question = prompt
            answer = get_answer(question)
            st.write(answer)
            # st.write(similar_docs)
            # st.write(answer1)
        except Exception:
            st.write("Documents loaded PRESS ASK again ")
            pass

























    

# Create a dictionary mapping page names to the corresponding functions
pages = {
    "Axstream Agents 🦜": home,
    "BABY AGI👼(websearch)": about,
    "CHAT💬-DOC📄": contact
}
st.sidebar.title("SELECT 🤖 BELOW")
# Add a sidebar to navigate between pages
page = st.sidebar.radio(".", options=list(pages.keys()))




with st.sidebar:
    openai_api_key =st.secrets[openai_api_key
    pineconekey =st.secrets[pineconekey]
    pineconeEnv ="us-west1-gcp-free"
    index_name2 ="axstream"
    serp_api =st.secrets[serp_api]




if openai_api_key and pineconekey and pineconeEnv and index_name2 and serp_api:

    st.success("!")
    pages[page]()

        # You can use the API keys in your code here
        # For example, make API requests using the keys

        # ...
else:
    st.warning("Please fill correct API keys .")



# Execute the function corresponding to the selected page
#pages[page]()








