import pandas as pd
import numpy as np
from langchain.chat_models import  ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import dotenv
# Embed and store splits
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
# it is used to generate progress bar we will import it with in loop.
from tqdm import tqdm
import os
from langchain.schema.runnable import RunnablePassthrough
import gradio as gr
import time


dotenv.load_dotenv()
class ZameenGPT:
    def __init__(self,model = 'gpt-3.5-turbo'):
        self.retriever = self.create_load_db()
        self.llm = ChatOpenAI(model_name = model, temperature = 0,)
        self.template = """Your Name is ZameenGPT. You are a Realestate Expert with knowledge of all the Properties in Islamabad, Pakistan.
        You will try to recommend the best property depending on the user's need from the given properties. 
        Always keep in mind the customer's requirements and suggest based on that.
        Keep your Answers structured and professional.
        Try to answer the user's property related question and try to recommend the best option for them.
        These are the available properties: 
        {context}

        This is the user's Question:
        {question}
        """
        self.rag_prompt = PromptTemplate.from_template(self.template)
        # RAG chain 
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()} 
            | self.rag_prompt 
            | self.llm 
        )
    def create_load_db(self):
        df=pd.read_csv('isl_data.csv')            
        data = df.drop(columns=['Unnamed: 0','property_id', 'location_id','type_id','price','city_id','province_id', 'province','purpose_id','date_added','agency_id','agent_id','amenities'])    
        # ist we use for loop then we use tqdm for progress bar 
        # it will check data.shape[0] index start from 0 until last one
        if not os.path.exists('db'):
            # creating vectordatabase with name 'db', then we embedding using OpenAIEmbeddings(), giving name 'zameen_db'
            vectorstore = Chroma(persist_directory='db',embedding_function=OpenAIEmbeddings(),collection_name='zameen_db')    
            for i in tqdm(range(data.shape[0])):
                data=df.iloc[[i]]
                zameen_string=f"""Property Type : {data['_type'][i]}
                Bedrooms: {data["bedrooms"][i]}
                Bathrooms: {data["baths"][i]}
                Location: {data["location"][i]}
                City: {data["city"][i]}
                Area(kanal): {data[" area(kanal)"][i]}
                URL : {data["page_url"][i]}
                Agency: {data["_agency"][i]}
                Price : {data["price_1"][i]}
                Latitude: {data["latitude"][i]}
                Longitude: {data["longitude"][i]}
                Agent: {data["_agent"][i]}
                Purpose: {data["_purpose"][i]}
                Description: {data["description"][i]}
                """
                vectorstore.add_texts([zameen_string])
                print(zameen_string)
            vectorstore.persist()
            vectorstore = None
        vectorstore = Chroma(persist_directory='db',embedding_function=OpenAIEmbeddings(),collection_name='zameen_db')
        retriever = vectorstore.as_retriever()
        return retriever
    def ask(self,question):
        print(self.retriever.invoke(question))
        response = self.rag_chain.invoke(question).content
        return response    

    
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        msg = gr.Textbox(container=False,scale=7,placeholder="whats your question")
        clear = gr.ClearButton()
    
    def respond(message,chat_history):
        bot_message = ZameenGPT().ask(message)
        chat_history.append((message,bot_message))
        time.sleep(1)
        return "",chat_history
    
    msg.submit(respond,[msg,chatbot],[msg,chatbot])
    
demo.queue()
demo.launch()