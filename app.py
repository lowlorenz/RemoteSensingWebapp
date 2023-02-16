"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import PIL.Image as Image

from evaluation import bleu, rouge, meteor


x = st.slider("Image Id", max_value=3149)

references = []
for i in range(5):
    with open(f"test/test_references_{i}.txt", "r") as f:
        references.append(f.read().split("\n")[x].replace(" .", ".")))

with open(f"test/test_hypothesis.txt", "r") as f:
    hypothesis = f.read().split("\n")[x]

with open(f"test/test_images.txt", "r") as f:
    image = f.read().split("\n")[x]

hypothesis_no_punc = hypothesis.replace(".", "")
references_no_punc = [ref.replace(" .", "") for ref in references]


bleu_score = bleu(references_no_punc, hypothesis_no_punc)
rouge_score = rouge(references_no_punc, hypothesis_no_punc)
meteor_score = meteor(references_no_punc, hypothesis_no_punc)


col1, col2, col3 = st.columns((1, 3, 1))

img = Image.open("test_images/" + image)
with col1:
    st.write(" ")

with col2:

    c1, c2, c3 = st.columns(3)
    c1.metric(label="BLEU Score", value=round(bleu_score, 3))
    c2.metric(label="ROUGE Score", value=round(rouge_score, 3))
    c3.metric(label="METEOR Score", value=round(meteor_score, 3))

    st.image(img, use_column_width=True)


with col3:
    st.write(" ")


with st.container():
    c1, c2, c3 = st.columns((1, 6, 1))
    c2.header("Hypthesis:")
    c2.write(hypothesis)
    c2.header("References:")
    for ref in references:
        c2.write(ref)
