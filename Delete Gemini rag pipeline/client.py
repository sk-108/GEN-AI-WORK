# import requests
# import streamlit as st

# def process_pdfs(files):
#     files = [('files', (file.name, file.read(), file.type)) for file in files]
#     response = requests.post("http://localhost:8000/process-pdf/", files=files)
#     return response.json()

# def ask_question(question):
#     response = requests.post("http://localhost:8000/ask-question/", json={"question": question})
#     response_data = response.json()
#     if 'answer' in response_data:
#         return response_data['answer']
#     else:
#         return response_data  # Return the whole response for debugging

# st.title('Chat with PDF using Gemini💁')

# uploaded_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=['pdf'])

# if st.button("Submit & Process") and uploaded_files:
#     with st.spinner("Processing..."):
#         result = process_pdfs(uploaded_files)
#         st.success(result['message'])

# user_question = st.text_input("Ask a Question from the PDF Files")

# if user_question:
#     answer = ask_question(user_question)
#     st.write("Reply: ", answer)

#second app

# import requests
# import streamlit as st

# def process_pdfs(files):
#     files = [('files', (file.name, file.read(), file.type)) for file in files]
#     response = requests.post("http://localhost:8000/process-pdf/", files=files)
#     return response.json()

# def ask_question(question):
#     response = requests.post("http://localhost:8000/ask-question/", json={"question": question})
#     try:
#         response.raise_for_status()  # Raise an error for bad responses
#         response_data = response.json()
#         if 'answer' in response_data:
#             return response_data['answer']
#         else:
#             return response_data  # Return the whole response for debugging
#     except requests.exceptions.RequestException as e:
#         st.error(f"Request failed: {e}")
#         st.error(f"Response content: {response.content}")

# st.title('Chat with PDF using LLM💁')

# uploaded_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=['pdf'])

# if st.button("Submit & Process") and uploaded_files:
#     with st.spinner("Processing..."):
#         result = process_pdfs(uploaded_files)
#         st.success(result['message'])

# user_question = st.text_input("Ask a Question from the PDF Files")

# if user_question:
#     answer = ask_question(user_question)
#     st.write("Reply: ", answer)

#third app
import requests
import streamlit as st

def process_pdfs(files):
    try:
        files = [('files', (file.name, file.read(), file.type)) for file in files]
        response = requests.post("http://localhost:8000/process-pdf/", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to process PDFs: {e}")
        return None

def ask_question(question):
    try:
        response = requests.post("http://localhost:8000/ask-question/", json={"question": question})
        response.raise_for_status()
        response_data = response.json()
        if 'answer' in response_data:
            return response_data['answer']
        else:
            st.error("No answer found in the response.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        st.error(f"Response content: {response.content}")
        return None

st.title('Chat with PDF using LLM💁')

uploaded_files = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=['pdf'])

if st.button("Submit & Process") and uploaded_files:
    with st.spinner("Processing..."):
        result = process_pdfs(uploaded_files)
        if result:
            st.success(result['message'])
        else:
            st.error("Failed to process the uploaded PDFs.")

user_question = st.text_input("Ask a Question from the PDF Files")

if user_question:
    with st.spinner("Getting the answer..."):
        answer = ask_question(user_question)
        if answer:
            st.write("Reply: ", answer)
        else:
            st.error("Failed to get an answer to your question.")
