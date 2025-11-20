content = '''fastapi
uvicorn
pymongo
python-multipart
pypdf
langchain
langchain-community
langchain-core
langchain-groq
pinecone
langchain-google-genai
python-dotenv
passlib[bcrypt]
streamlit
requests
PyPDF2
tqdm
'''
with open(r'd:\Github\medicalDiagnosis\requirements.txt','w',encoding='utf-8') as f:
    f.write(content)
print('wrote requirements.txt')
