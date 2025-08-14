import streamlit as st
from langchain import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Título de la app
st.title("🌱 Agente de Agricultura con LangChain y HuggingFace")

# Entrada del usuario
query = st.text_input("Pregunta sobre agricultura:")

# HuggingFace API key (poner en Streamlit Secrets)
HF_API_KEY = st.secrets["HF_API_KEY"]

if query:
    try:
        # Configurar modelo desde HuggingFace
        llm = HuggingFaceHub(
            repo_id="tiiuae/falcon-7b-instruct",
            huggingfacehub_api_token=HF_API_KEY,
            model_kwargs={"temperature": 0.5, "max_length": 256}
        )

        # Crear prompt
        template = """
        Eres un experto en agricultura. Responde la siguiente pregunta de forma clara y breve:
        {pregunta}
        """
        prompt = PromptTemplate(input_variables=["pregunta"], template=template)
        chain = LLMChain(llm=llm, prompt=prompt)

        # Obtener respuesta
        respuesta = chain.run(pregunta=query)
        st.success(respuesta)

    except Exception as e:
        st.error(f"Error: {e}")
