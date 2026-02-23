import streamlit as st
import os
import secrets
from pdf_para_dxf import converter_pdf_para_dxf

st.set_page_config(page_title="CONVERSOR PDF PARA DWG", page_icon="🏗️", layout="centered")

# Custom CSS for a premium look
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #006AFF;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056D2;
        border: none;
        color: white;
    }
    h1 {
        color: white;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("CONVERSOR PDF PARA DWG")
st.write("---")

uploaded_file = st.file_uploader("Arraste seu PDF aqui ou clique para selecionar", type=['pdf'])

if uploaded_file is not None:
    # Use a secure temp name
    token = secrets.token_hex(4)
    temp_pdf = f"temp_{token}_{uploaded_file.name}"
    temp_dwg = temp_pdf.rsplit('.', 1)[0] + ".dwg"
    
    # Save the uploaded file temporarily
    with open(temp_pdf, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info("Convertendo arquivo... Isso pode levar alguns segundos.")
    
    try:
        # Perform conversion
        converter_pdf_para_dxf(temp_pdf, temp_dwg)
        
        if os.path.exists(temp_dwg):
            st.success("Conversão concluída com sucesso!")
            
            # Read converted file for download
            with open(temp_dwg, "rb") as file:
                btn = st.download_button(
                    label="Baixar Arquivo DWG",
                    data=file,
                    file_name=uploaded_file.name.rsplit('.', 1)[0] + ".dwg",
                    mime="application/octet-stream"
                )
        else:
            st.error("Erro na conversão. O arquivo DWG não pôde ser gerado.")
            
    except Exception as e:
        st.error(f"Erro no processamento: {str(e)}")
    
    finally:
        # Cleanup
        if os.path.exists(temp_pdf):
            os.remove(temp_pdf)
        if os.path.exists(temp_dwg):
            os.remove(temp_dwg)

st.write("---")
st.caption("Ferramenta de conversão rápida | Fidelidade Total de Cores e Polígonos")
