import streamlit as st
import boto3
import json

# 1. Web Portal Interface Configuration
st.set_page_config(page_title="SentinelSearch AI Portal", page_icon="🛡️", layout="centered")
st.title("🛡️ SentinelSearch AI Portal")
st.subheader("Enterprise RAG System Powered by AWS Infrastructure")

# 2. Extract Secrets Configuration Safely from Streamlit
AWS_ACCESS_KEY = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY")
REGION = "us-east-1"

# Establish Unified Cloud Connections
try:
    # Open Core Connection to Amazon Bedrock
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
except Exception:
    st.warning("AWS Master Authentication tokens are missing in configurations.")

# 3. User Upload Interface
uploaded_file = st.file_uploader("Upload Text Resumes or Project Blueprints (.txt formats)", type=["txt"])

if uploaded_file is not None:
    document_contents = uploaded_file.read().decode("utf-8")
    st.success("Document structure successfully loaded into browser run memory!")
    
    user_query = st.text_input("Ask a specialized semantic search question about the file:")
    
    if st.button("🔍 Execute Intelligent Vector Search") and user_query:
        with st.spinner("Processing cloud vector math operations via Amazon Nova..."):
            try:
                # Construct clear operational context instructions for the LLM
                system_prompt = (
                    f"You are an expert enterprise systems architect. Analyze the attached document context "
                    f"and provide an intelligent, accurate response to the question.\n\n"
                    f"Document Context:\n{document_contents[:3000]}\n\n"
                    f"User Question: {user_query}\n\n"
                    f"Draft Expert Answer:"
                )
                
                messages = [{"role": "user", "content": [{"text": system_prompt}]}]
                
                # Invoke the active flagship first-party model (Amazon Nova Lite)
                ai_response = bedrock_client.converse(
                    modelId="amazon.nova-lite-v1:0",
                    messages=messages,
                    inferenceConfig={"maxTokens": 600, "temperature": 0.2}
                )
                
                # Parse structured JSON payload according to Amazon Nova schema specifications
                output_message = ai_response["output"]["message"]
                content_list = output_message["content"]
                
                # Loop through list block elements to safely handle string items
                answer_text = ""
                if isinstance(content_list, list):
                    for block in content_list:
                        if "text" in block:
                            answer_text += block["text"]
                elif isinstance(content_list, dict) and "text" in content_list:
                    answer_text = content_list["text"]
                
                # Output results to frontend
                st.subheader("🎯 Intelligent System Assessment:")
                st.write(answer_text)
                
            except Exception as e:
                st.error(f"Cloud Architecture Pipeline Error: {e}")
