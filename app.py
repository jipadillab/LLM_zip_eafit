import streamlit as st
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Configuración de la página
st.set_page_config(page_title="Agente de Agricultura", page_icon="🌱")
st.title("🌱 Agente de Agricultura con LangChain y HuggingFace")
st.markdown("Pregunta sobre cultivos, fertilización, enfermedades de plantas y buenas prácticas agrícolas.")

# Token de HuggingFace (configurado en Streamlit Secrets)
HF_API_KEY = st.secrets["HF_API_KEY"]

# Inicializar el modelo
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",  # modelo de ejemplo
    huggingfacehub_api_token=HF_API_KEY,
    model_kwargs={"temperature": 0.5, "max_length": 256}
)

# Plantilla de prompt
template = """
Eres un experto en agricultura. Responde la siguiente pregunta de forma clara y breve.
Pregunta: {pregunta}
Respuesta:
"""
prompt = PromptTemplate(input_variables=["pregunta"], template=template)
chain = LLMChain(llm=llm, prompt=prompt)

# Entrada del usuario
query = st.text_input("Escribe tu pregunta:")

# Botón para generar respuesta
if st.button("Obtener respuesta"):
    if query.strip() == "":
        st.warning("Por favor escribe una pregunta.")
    else:
        try:
            respuesta = chain.run(pregunta=query)
            st.success(respuesta)
        except Exception as e:
            st.error(f"Error al generar la respuesta: {e}")

# Ejemplos de preguntas para probar
st.markdown("**Ejemplos de preguntas:**")
st.markdown("""
- ¿Cuáles son las mejores prácticas para fertilizar maíz?
- ¿Cómo detectar y controlar la roya en el café?
- ¿Qué cultivos son recomendables en suelos ácidos?
- ¿Cuándo es el mejor momento para sembrar trigo en clima templado?
- ¿Qué enfermedades afectan al tomate y cómo prevenirlas?
""")
