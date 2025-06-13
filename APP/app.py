# Specify the file path
file_path = "APP/warpeace_input.txt"
# Open and read the contents of the .txt file
with open(file_path, 'r') as file:
    data = file.read()
# Display the content
print(len(data))
import re
# Define the regular expression to clean each line
def clean_line(line):
    # Modify this regex depending on the type of text/code you're processing
    return re.sub('[^a-zA-Z0-9 \.]', '', line)

# File paths
  # Input text file with special characters
output_file = 'cleaned_text.txt'  # Output file to write cleaned content

# Open the input file and clean each line
with open(file_path, 'r') as file:
    # Open output file to write cleaned lines
    with open(output_file, 'w') as cleaned_file:
        for line in file:
            # Clean the current line using the clean_line function
            cleaned_line = clean_line(line)
            # Write the cleaned line to the new file
            cleaned_file.write(cleaned_line + '\n')

print(f"Text cleaned and saved to {output_file}")
with open('cleaned_text.txt', 'r') as file:
    cleaned_data = file.read()
output_file = 'lower_cased_cleaned_text.txt'
with open('cleaned_text.txt', 'r') as file:
    # Open output file to write cleaned lines
    with open(output_file, 'w') as cleaned_file:
        for line in file:
            # Clean the current line using the clean_line function
            cleaned_file.write(line.lower())
unique_words = set()
lines_dataset = []
with open('lower_cased_cleaned_text.txt', 'r') as file:
    lines = file.readlines()
    lines_dataset = lines
    for line in lines:
        unique_words.update(line.split())

# Assuming 'words' is a list of unique words you obtained from the previous step
words = sorted(list(set(unique_words)))  # unique_words is from the previous processing

# Create mappings for words to integers and vice versa
stoi = {word: i + 1 for i, word in enumerate(words)}  # Word to index mapping
stoi['<PAD>'] = 0  # Optional: Add a padding token if needed
itos = {i: word for word, i in stoi.items()}  # Index to word mapping

import streamlit as st
import torch
import re

# Load model and dictionaries (replace these with actual paths)
# checkpoint = torch.load('path/to/your/model.pth', map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()  # Set the model to evaluation mode

# Example stoi and itos dictionaries (replace with actual ones)
# stoi = {'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5}
# itos = {v: k for k, v in stoi.items()}

# Define your model, or load a pretrained model
# model = YourModelClass()
import torch.nn as nn
import torch.nn.functional as F  # Missing import

# Define the model for word-level generation with hidden layers
class NextWord(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size=1024):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)  # Output layer

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)  # Flatten the embeddings
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

# Define device for computation (CUDA if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
block_size = 5  # Adjust this according to your use case
emb_dim = 64    # Set the embedding dimension

# Assuming `stoi` and `itos` are predefined vocab mappings
# Example: stoi = {'<pad>': 0, '<end>': 1, 'word1': 2, ...}, itos = {0: '<pad>', 1: '<end>', 2: 'word1', ...}
vocab_size = len(stoi)
model = NextWord(block_size, vocab_size, emb_dim).to(device)

# If using torch >= 2.0, compile the model (remove this if not using torch 2.x or later)
# model = torch.compile(model)

# Seed for reproducibility
g = torch.Generator()
g.manual_seed(4000002)

# Function to generate names based on words
def generate_name_words(model, itos, stoi, block_size, max_len=10):
    context = [0] * block_size  # Start with context filled with zeros (padding)
    name = []
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)  # Prepare input for the model
        y_pred = model(x)  # Get predictions
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()  # Sample from predictions
        word = itos[ix]  # Get the word corresponding to the sampled index
        if word == '<end>':  # Define an end-of-sequence token
            break
        name.append(word)
        context = context[1:] + [ix]  # Update the context by appending the new word index
    return ' '.join(name)  # Join the words to form a nam

# Load the model
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# Define the model class (same as before)
class NextWord(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size=1024):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)  # Output layer

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)  # Flatten the embeddings
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

# Define device for computation (CUDA if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load multiple models
def load_models():
    model1 = NextWord(block_size, vocab_size, emb_dim).to(device)
    checkpoint1 = torch.load('model.pth', map_location='cpu')
    model1.load_state_dict(checkpoint1['model_state_dict'])

    model2 = NextWord(block_size, vocab_size, emb_dim).to(device)
    checkpoint2 = torch.load('model11.pth', map_location='cpu')
    model2.load_state_dict(checkpoint2['model_state_dict'])

    return model1, model2

# Function to preprocess input
def preprocess_input(sentence, stoi, block_size):
    sentence = sentence.lower()  # Convert to lowercase
    sentence = re.sub(r'[^a-z\s\.]', '', sentence)  # Remove special characters except full stops
    words = sentence.split()  # Split into words
    words = [word for word in words if word]  # Remove empty strings

    # Create context from the last block_size words
    context = [0] * block_size  # Start with the end-of-sentence token
    for i in range(block_size):
        if i < len(words):
            word = words[i]
            context[i] = stoi.get(word, 0)  # Get the index, use 0 if word not in vocab
        else:
            context[i] = 0  # Fill with end-of-sentence token if fewer words

    return context

# Predict next k words
def predict_next_k_words(model, input_indices, k, itos):
    model.eval()  # Set the model to evaluation mode
    context = input_indices
    context_tensor = torch.tensor(context, dtype=torch.long).view(1, -1).to(device)
    predicted_words = []

    for i in range(k):
        with torch.no_grad():
            y_pred = model(context_tensor)  # Forward pass
            ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
            next_word = itos[ix]
            context_tensor = torch.tensor(list(context_tensor[0][1:].numpy()) + [ix], dtype=torch.long).view(1, -1).to(device)
            predicted_words.append(next_word)
    
    return predicted_words

# Load models
model1, model2 = load_models()

# Streamlit App
st.title("Next Word Prediction App with Multiple Models")

# Let the user choose a model
model_choice = st.selectbox("Choose a model:", ["Model 1", "Model 2"])
activation_function = st.selectbox("Choose activation function:", ["ReLU", "Tanh"])
embedding_size = st.selectbox("Choose embedding size:", ["128", "64"])

# User input
input_text = st.text_input("Enter initial text for prediction:", "we can say that")
context_length = 5
max_len = st.slider("Number of words to predict", 1, 10, 5)

# Button to generate text
if st.button("Generate Text"):
    input_indices = preprocess_input(input_text, stoi, context_length)

    # Choose the appropriate model based on user selection
    if model_choice == "Model 1":
        generated_text = predict_next_k_words(model1, input_indices, max_len, itos)
    elif model_choice == "Model 2":
        generated_text = predict_next_k_words(model2, input_indices, max_len, itos)

    # Display the generated text
    st.write("Generated text:", input_text + " " + " ".join(generated_text))

