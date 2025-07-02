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



# MLP Model
class NextWordMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate, context_size, activation_function):
        super(NextWordMLP, self).__init__()
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

# LSTM Model
class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate):
        super(NextWordLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        final_output = output[:, -1, :]
        final_output = self.dropout(final_output)
        logits = self.fc(final_output)
        return F.log_softmax(logits, dim=1)

def clean_and_tokenize(text):
    if not text.strip():
        return []
    text = text.replace('start', ' start ').replace('end', ' end ')
    text = re.sub(r'([.,!?])', r' \1 ', text)
    return re.sub(r'\s+', ' ', text).strip().lower().split()

def words_to_indices(words):
    return [word_to_index.get(word, word_to_index['pad']) for word in words]



# Model loader for MLP
def load_mlp_model(context_size, embedding_dim, activation_fn, random_seed):
    model = NextWordMLP(vocab_size, embedding_dim, hidden_dim=1024, dropout_rate=0.3,
                        context_size=context_size, activation_function=activation_fn).to(device)
    model_path = f'models/mlp/model_context_{context_size}_emb_{embedding_dim}_act_{activation_fn.__name__}_seed_{random_seed}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Model loader for LSTM
def load_lstm_model(context_size, embedding_dim, random_seed):
    model = NextWordLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=512,
        num_layers=2,
        dropout_rate=0.3
    ).to(device)
    model_path = f'models/lstm/LSTM_context_{context_size}_emb_{embedding_dim}_layers_2_seed_{random_seed}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# Text generation for MLP
def generate_text_mlp(model, start_sequence, num_words, context_size, random_seed, temperature=1.0):
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
        if index_to_word[next_word_idx] == 'end':
            break
    generated_text = ' '.join(index_to_word[idx] for idx in generated if index_to_word[idx] != 'pad')
    sentences = generated_text.split('. ')
    sentences = [s for s in sentences if s not in ['start', 'end']] # dont display the words start and end
    formatted_sentences = [s.capitalize() for s in sentences]   # capitalize the first letter of each sentence
    formatted_text = '. '.join(formatted_sentences)
    if not formatted_text.endswith('.'):    # add period at the end if missing
        formatted_text += '.'
    return formatted_text

# Text generation for LSTM
def generate_text_lstm(model, start_sequence, num_words, context_size, random_seed, temperature=1.0):
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
        if index_to_word[next_word_idx] == 'end':
            break
    generated_text = ' '.join(index_to_word[idx] for idx in generated if index_to_word[idx] != 'pad')
    sentences = generated_text.split('. ')
    sentences = [s for s in sentences if s not in ['start', 'end']] # dont display the words start and end
    formatted_sentences = [s.capitalize() for s in sentences]   # capitalize the first letter of each sentence
    formatted_text = '. '.join(formatted_sentences)
    if not formatted_text.endswith('.'):    # add period at the end if missing
        formatted_text += '.'
    return formatted_text


def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)





st.title("Next Word Generator")
st.write("Generate text using a next-word generation model.")


# Model type selection with buttons
model_type = st.radio(
    "Choose Model Type:",
    options=["MLP", "LSTM"],
    index=["MLP", "LSTM"].index(st.session_state.get("model_type", "MLP")),
    horizontal=True
)
st.session_state['model_type'] = model_type


# Improved seed text logic
default_seed_prompts = [
    "Mix milk and cream in a bowl",
    "Chop onions and tomatoes finely",
    "Boil rice until soft and fluffy",
]


if 'seed_text' not in st.session_state:
    st.session_state['seed_text'] = random.choice(default_seed_prompts)
input_text = st.text_input("Enter the starting sequence of words:", value=st.session_state['seed_text'])

random_seed = st.slider("Random Seed", min_value=0, max_value=100, value=42, step=1)
temperature = st.slider("Temperature", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
num_words = st.slider("Number of Words to Generate", min_value=10, max_value=200, value=100, step=5)

if model_type == "MLP":
    context_size = st.selectbox("Context Size", [5, 10], index=1)
    embedding_dim = st.selectbox("Embedding Dimension", [32, 64], index=0)
    activation_fn_name = st.selectbox("Activation Function", ["tanh", "leaky_relu"], index=0)
    activation_fn_map = {"tanh": torch.tanh, "leaky_relu": F.leaky_relu}
    activation_fn = activation_fn_map[activation_fn_name]
    model = load_mlp_model(context_size, embedding_dim, activation_fn, 42)
else:
    context_size = st.selectbox("Context Size", [5, 10, 20, 40], index=1)
    embedding_dim = st.selectbox("Embedding Dimension", [32, 64, 128], index=0) 
    model = load_lstm_model(context_size, embedding_dim, 0)

start_sequence_words = clean_and_tokenize(input_text)
start_sequence_indices = words_to_indices(start_sequence_words)
if len(start_sequence_indices) < context_size:
    start_sequence_indices = [word_to_index['pad']] * (context_size - len(start_sequence_indices)) + start_sequence_indices

if st.button("Generate Text"):
    if model_type == "MLP":
        generated_text = generate_text_mlp(model, start_sequence_indices, num_words, context_size, random_seed, temperature)
    else:
        generated_text = generate_text_lstm(model, start_sequence_indices, num_words, context_size, random_seed, temperature)

    st.write("Generated Text:")
    output_placeholder = st.empty()
    accumulated_text = ""
    for word in stream_data(generated_text):
        accumulated_text += word
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

