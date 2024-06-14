# Policy Document QA Chatbot

This project is a Policy Document QA Chatbot built using Streamlit, Hugging Face Transformers, and LangChain. The chatbot extracts information from a policy document PDF and answers user queries about the policy. It includes features like typing animation and follow-up question functionality.

## Tasks Performed

- **Extract Text from PDF**: Extracts text content from the provided policy PDF document.
- **Text Chunking**: Splits the extracted text into manageable chunks for processing.
- **Vector Store Creation**: Uses embeddings to create a FAISS vector store for efficient document retrieval.
- **Question Answering**: Generates responses to user queries using a Hugging Face language model.
- **Typing Animation**: Simulates typing animation for generating responses.
- **Follow-Up Questions**: Allows users to ask follow-up questions and get relevant answers.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/policy-document-qa-chatbot.git
    cd policy-document-qa-chatbot
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the necessary language model:
    ```python
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    generator_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    ```

## Usage

1. Place your policy document PDF in the same directory as the script and name it `policy-booklet-0923.pdf`.

2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

3. Open the web browser and navigate to `http://localhost:8501` to interact with the chatbot.

## Code Overview

### `app.py`

This is the main script that runs the Streamlit app. It includes:

- **PDF Text Extraction**: Extracts text from the provided PDF document.
- **Text Chunking**: Splits the text into chunks for easier processing.
- **Vector Store Creation**: Creates a FAISS vector store using embeddings.
- **Question Answering**: Uses a Hugging Face language model to generate responses to user queries.
- **Streamlit UI**: Provides an interactive UI for users to ask questions and view responses.

### `requirements.txt`

Lists all the required Python packages for the project:
```
streamlit
PyPDF2
transformers
sentence-transformers
langchain
faiss-cpu
streamlit-chat
```

## Example

Here’s how to interact with the chatbot:

1. Start the app:
    ```bash
    streamlit run app.py
    ```

2. Enter a question in the input box, such as "What is the cancellation policy?".

3. The chatbot will display a typing animation and then provide an answer based on the policy document.

4. You can also ask follow-up questions to get more detailed information.

## Future Improvements

- **Advanced Models**: Integrate more advanced language models to improve response accuracy.
- **Fine-Tuning**: Fine-tune the language model on specific datasets for better performance.
- **Enhanced UI**: Add more interactive elements and improve the overall user experience.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue if you have any suggestions or bug reports.

## License

This project is licensed under the MIT License.
