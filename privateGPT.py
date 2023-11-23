#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings #mychange
from langchain.memory import ConversationBufferMemory #memchange
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
import pandas as pd
import re
import glob
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')

from constants import CHROMA_SETTINGS

def handle_csv_query(query: str, source_directory: str):
    
    # Add a list of keywords related to CSV operations
    csv_keywords = ['sum', 'average', 'max', 'min', 'count']

    # Check if the query contains any CSV-related keywords
    if not any(keyword in query.lower() for keyword in csv_keywords):
        return None, None
    
    pattern = r"(?i)(average|max|min|sum) (?:number of|of|in) ([\w\s]+)"
    match = re.search(pattern, query)

    if not match:
        # Check for row and column count queries
        row_patterns = [
            r"(?i)how many rows (?:are there in|in) (?:the )?([\w\s]+\.csv)",
            r"(?i)tell me (?:the )?rows (?:of|in) (?:the )?([\w\s]+\.csv)",
            r"(?i)what is the number of rows (?:of|in) (?:the )?([\w\s]+\.csv)",
            r"(?i)count (?:me )?the number of rows (?:in|of) (?:the )?([\w\s]+\.csv)"
        ]

        col_patterns = [
            r"(?i)how many columns (?:are there in|in) (?:the )?([\w\s]+\.csv)",
            r"(?i)tell me (?:the )columns (?:|in) (?:the )?([\w\s]+\.csv)",
            r"(?i)what is the number of columns (?:of|in) (?:the )?([\w\s]+\.csv)",
            r"(?i)count (?:me )the columns (?:in|of) (?:the )?([\w\s]+\.csv)"
        ]

        row_match = None
        col_match = None

        for pattern in row_patterns:
            row_match = re.search(pattern, query)
            if row_match:
                break

        for pattern in col_patterns:
            col_match = re.search(pattern, query)
            if col_match:
                break

        if not row_match and not col_match:
            return None,None

    if match:
        operation, column_name = match.groups()
        operation = operation.lower()
    else:
        operation = None

    # Load all CSV files
    csv_files = glob.glob(os.path.join(source_directory, "**/*.csv"), recursive=True)
    if not csv_files:
        return "No CSV files found.", None

    # Find the CSV file containing the column name mentioned in the query
    target_file = None
    target_df = None
    target_column = None

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if operation:
            column_matches = process.extractOne(column_name, df.columns, scorer=fuzz.token_set_ratio)
            best_match, score = column_matches

            if score >= 50:
                target_file = csv_file
                target_df = df
                target_column = best_match
                break
        else:
            if row_match or col_match:
                target_file = csv_file
                target_df = df
                break

    if not target_file:
        if operation:
            return f"Column '{column_name}' not found in any CSV file. Please check the column name.", None
        else:
            return "No matching CSV file found.", None

    if operation:
        if operation == "average":
            result = target_df[target_column].mean()
        elif operation == "max":
            result = target_df[target_column].max()
        elif operation == "min":
            result = target_df[target_column].min()
        elif operation == "sum":
            result = target_df[target_column].sum()
        else:
            return "Unsupported operation.", None

        return f"The {operation} of '{target_column}' is {result:.2f}.", target_file
    else:
        if row_match:
            return f"There are {len(target_df)} rows in the CSV file.", target_file
        elif col_match:
            return f"There are {len(target_df.columns)} columns in the CSV file.", target_file


class CustomRetrievalQA(RetrievalQA):
    def prep_outputs(self, inputs, outputs, return_only_outputs):
        self._validate_outputs(outputs)
        if self.memory is not None:
            # Modify the outputs dictionary to include only the 'result' key
            modified_outputs = {'result': outputs['result']}
            self.memory.save_context(inputs, modified_outputs)
        if return_only_outputs:
            return outputs[self.output_key]
        return outputs       
        
def main():
    # Parse the command line arguments
    args = parse_arguments()
#     embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_model_name) #mychange
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    
    #defining memory
    memory = ConversationBufferMemory()   #memchange
    # Prepare the LLM
    if model_type == "GPT4All":
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
    else:
        print(f"Model {model_type} not supported!")
        exit()
    qa = CustomRetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source, memory=memory)
    # Interactive questions and answers
    print(f"Source directory: {source_directory}")

    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        
        # Check if the query is related to a calculation on a CSV file
        csv_result, source_file = handle_csv_query(query, source_directory)
        if csv_result:
            print("\n> Answer:")
            print(csv_result)
            if source_file:
                print(f"Source File: {source_file}")
            memory.save_context({"query": query}, {"result": csv_result})
            print("Memory:",memory)
            continue
        
        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\nSource File: " + document.metadata["source"] + ":")
            print(document.page_content)
        
        print("Memory:",memory)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
