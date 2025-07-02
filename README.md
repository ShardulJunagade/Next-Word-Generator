# Next Word Generator

This project implements a Next-Word Generator using both MLP and LSTM neural network architectures. All models are trained on the [Cleaned Indian Recipes Dataset](https://www.kaggle.com/datasets/sooryaprakash12/cleaned-indian-recipes-dataset) from Kaggle. 

It also features an interactive Streamlit web app for text generation, which is also deployed at [Recipe Next Word Generator](https://recipe-next-word-generator.streamlit.app/).

## Features
- **MLP and LSTM Models:** Choose between Multi-Layer Perceptron and LSTM for next-word generation.
- **Interactive Streamlit App:** User-friendly interface for generating text based on a seed prompt with customizable parameters like context size, embedding dimension, activation function, random seed and temperature.
- **Pre-trained Models:** No need to retrain; select from pre-trained variants.
- **Word Embedding Visualization:** Notebooks for t-SNE and other embedding visualizations.




## Project Structure

```
├── assets/
├── models/
│   ├── mlp/
│   └── lstm/
├── streamlit_app.py
├── MLP.ipynb
├── LSTM.ipynb
├── embeddings.ipynb
├── README.md
└── requirements.txt
```

## Getting Started


### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ShardulJunagade/Next-Word-Generator.git
   cd Next-Word-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install uv
   uv pip install -r requirements.txt
   ```

### Running the Streamlit App
```bash
streamlit run streamlit_app.py
```
The app will open in your browser. Enter a seed text or use the default, select model and parameters, and generate text.

## Notebooks
- [`MLP.ipynb`](./MLP.ipynb): Data processing, training, and evaluation for the MLP model.
- [`LSTM.ipynb`](./LSTM.ipynb): Data processing, training, and evaluation for the LSTM model.
- [`embeddings.ipynb`](./embeddings.ipynb): Visualization of learned word embeddings.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.