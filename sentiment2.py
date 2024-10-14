import tkinter as tk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Load the model and tokenizer
model = load_model("Sentiment_Analysis_on_TR")
with open('tokenize.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def classify_sentiment():
    content = entry_content.get("1.0", tk.END).strip()

    if not content:
        result_text.set("Please enter some content for analysis.")
        return

    # Preprocess content and perform text summarization
    text_ = content.lower()
    Stopwords = set(stopwords.words('english'))
    word_freq = {}
    for word in word_tokenize(text_):
        if word not in Stopwords:
            if word not in word_freq.keys():
                word_freq[word] = 1
            else:
                word_freq[word] += 1
    max_freq = max(word_freq.values())

    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq

    sent_list = sent_tokenize(text_)
    sent_scores = {}
    for sent in sent_list:
        for word in word_tokenize(sent.lower()):
            if word in word_freq.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sent_scores.keys():
                        sent_scores[sent] = word_freq[word]
                    else:
                        sent_scores[sent] += word_freq[word]
    summary_sentences = heapq.nlargest(5, sent_scores, key=sent_scores.get)  # Taking the top 5 sentences
    Cap_sent = []
    for tex in summary_sentences:
        tex_cap = tex.capitalize()
        Cap_sent.append(tex_cap)
    summary = " ".join(Cap_sent)

    # Sentiment analysis
    text_list = [content]
    text_token = tokenizer.texts_to_sequences(text_list)
    text_pad = pad_sequences(text_token, maxlen=241, padding='pre')
    pred = model.predict(text_pad)
    if pred[0][0] > 0.5:
        result = "Positive Review!"
        per = round((pred[0][0]) * 100, 2)
    else:
        result = "Negative Review!"
        per = round((pred[0][0]) * 100, 2)

    # Clear the previous result before displaying the new one
    result_display.delete("1.0", tk.END)

    # Displaying the result in the GUI
    result_display.insert(tk.END, f"Content Analyzed:\n{content}\n\nSummary:\n{summary}\n\n"
                                  f"Sentiment: {result}\nCertainty: {per}%\n")

    # Clear input after classification for the next entry
    entry_content.delete("1.0", tk.END)

# Tkinter GUI setup
root = tk.Tk()
root.title("Interactive Sentiment Analysis Tool")

# Text input for content
tk.Label(root, text="Enter Content:").grid(row=0, column=0, padx=10, pady=10)
entry_content = tk.Text(root, height=10, width=60)
entry_content.grid(row=0, column=1, padx=10, pady=10)

# Classify button
btn_classify = tk.Button(root, text="Analyze Sentiment", command=classify_sentiment)
btn_classify.grid(row=1, column=0, columnspan=2, pady=20)

# Result display
tk.Label(root, text="Analysis Results:").grid(row=2, column=0, padx=10, pady=10)
result_display = tk.Text(root, height=15, width=60)
result_display.grid(row=2, column=1, padx=10, pady=10)

# Scrollbar for results
scrollbar = tk.Scrollbar(root, command=result_display.yview)
scrollbar.grid(row=2, column=2, sticky='nsew')
result_display.config(yscrollcommand=scrollbar.set)

root.mainloop()
