import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os
from torch import nn
import streamlit as st
import json
import re
import random
import time 

with open("assets/word_to_index.json", "r") as f:
    word_to_index = json.load(f)

with open("assets/index_to_word.json", "r") as f:
    index_to_word = json.load(f)
    index_to_word = {int(k): v for k, v in index_to_word.items()}

vocab_size = len(word_to_index)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImprovedMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate, context_size, activation_function):
        super(ImprovedMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * context_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.embedding(x).view(x.size(0), -1)
        x = self.dropout1(self.activation_function(self.bn1(self.fc1(x))))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def clean_and_tokenize(text):
    if not text.strip():
        return []
    text = text.replace('start', ' start ').replace('end', ' end ')
    text = re.sub(r'([.,!?])', r' \1 ', text)
    return re.sub(r'\s+', ' ', text).strip().lower().split()

def words_to_indices(words):
    return [word_to_index.get(word, word_to_index['pad']) for word in words]


def load_model(context_size, embedding_dim, activation_fn, random_seed):
    model = ImprovedMLP(vocab_size, embedding_dim, hidden_dim=1024, dropout_rate=0.3,
                        context_size=context_size, activation_function=activation_fn).to(device)
    model_path = f'models/model_context_{context_size}_emb_{embedding_dim}_act_{activation_fn.__name__}_seed_{random_seed}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def generate_text(model, start_sequence, num_words, random_seed, temperature=1.0):
    torch.manual_seed(random_seed)
    model.eval()
    generated = list(start_sequence)
    for _ in range(num_words):
        input_seq = torch.tensor(generated[-context_size:], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_seq)
        logits = output.squeeze(0) / temperature
        next_word_idx = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item()
        generated.append(next_word_idx)

    
    generated_text = ' '.join(index_to_word[idx] for idx in generated if index_to_word[idx] != 'pad')
    sentences = generated_text.split('. ')
    
    sentences = [s for s in sentences if s not in ['start', 'end']]# don't display the words start and end
    formatted_sentences = [s.capitalize() for s in sentences]  # Capitalize first letter of each sentence
    formatted_text = '. '.join(formatted_sentences)

    if not formatted_text.endswith('.'): # Add period at the end if missing
        formatted_text += '.'
        
    return formatted_text


def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)


st.title("Next Word Prediction App")
st.write("Generate text using a next-word prediction MLP model.")


default_seed_length = st.slider("Default Seed Text Length", min_value=1, max_value=5, value=10, step=1)
default_seed_text = " ".join(index_to_word[idx] for idx in random.sample(range(vocab_size), default_seed_length))


input_text_option = st.radio("Choose seed text input method:", ( "Custom Seed Text","Default Seed Text"))
if input_text_option == "Default Seed Text":
    input_text = default_seed_text
else:
    input_text = st.text_input("Enter the starting sequence of words:", value="Mix milk and cream")

context_size = st.selectbox("Context Size", [5, 10], index=1)
embedding_dim = st.selectbox("Embedding Dimension", [32, 64, 128], index=1)
activation_fn_name = st.selectbox("Activation Function", ["tanh", "leaky_relu"], index=0)
random_seed = st.slider("Random Seed", min_value=0, max_value=100, value=42, step=1)
temperature = st.slider("Temperature", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
num_words = st.slider("Number of Words to Generate", min_value=10, max_value=200, value=100, step=5)

activation_fn_map = {"tanh": torch.tanh, "leaky_relu": F.leaky_relu}
activation_fn = activation_fn_map[activation_fn_name]


model = load_model(context_size, embedding_dim, activation_fn, 42)


start_sequence_words = clean_and_tokenize(input_text)
start_sequence_indices = words_to_indices(start_sequence_words)

if len(start_sequence_indices) < context_size:
    start_sequence_indices = [word_to_index['pad']] * (context_size - len(start_sequence_indices)) + start_sequence_indices

if st.button("Generate Text"):
    generated_text = generate_text(model, start_sequence_indices, num_words, random_seed, temperature)
    

    st.write("Generated Text:")
    output_placeholder = st.empty()

    accumulated_text = ""
    for word in stream_data(generated_text):
        accumulated_text += word  # Accumulate the generated text
        output_placeholder.markdown(accumulated_text, unsafe_allow_html=True)

    st.sidebar.subheader("Seed Text")
    seed_output_placeholder = st.sidebar.empty()

    accumulated_seed_text = ""
    for word in stream_data(input_text):
        accumulated_seed_text += word
        seed_output_placeholder.markdown(accumulated_seed_text, unsafe_allow_html=True)

    st.sidebar.header("Generated Text")
    generated_sidebar_placeholder = st.sidebar.empty()

    accumulated_generated_text = ""
    for word in stream_data(generated_text):
        accumulated_generated_text += word
        generated_sidebar_placeholder.markdown(accumulated_generated_text, unsafe_allow_html=True)

