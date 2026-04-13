import re
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


REPLY_PATTERN = r"(-----Original Message-----|From: .*\nSent: .*\nTo: .*\nSubject: .*)"
def strip_reply_chain(text):
    parts = re.split(REPLY_PATTERN, text, flags=re.IGNORECASE)
    return parts[0].strip()

SIGNATURE_MARKERS = [
    r"Regards,",
    r"Best,",
    r"Sincerely,",
    r"Thank you,",
    r"This e-mail.*?confidential",
    r"-----Original Message-----",
    r"From: .*@.*",
    r"Sent: .*",
    r"Phone: .*",
    r"Fax: .*",
    r"www\..*",
    r"\d{3}-\d{3}-\d{4}",  # phone numbers
]

def remove_signatures(text):
    for marker in SIGNATURE_MARKERS:
        text = re.split(marker, text, flags=re.IGNORECASE)[0]
    return text.strip()

def preprocess_emails(df, text_column="text", subject_column="subject"):
    df = df.copy()

    # combine subject + body
    df[text_column] = df[text_column].fillna("").astype(str)
    df[subject_column] = df[subject_column].fillna("").astype(str)
    df["raw_text"] = df[subject_column] + " " + df[text_column]

    df["clean_text"] = (
        df["raw_text"]
        .apply(strip_reply_chain)
        .apply(remove_signatures)
        .str.lower()
    )

    #parallel tokenization
    texts = df["clean_text"].tolist()

    tokens_list = []
    batch_size = 200
    n_process = 4   # use 4 CPU cores bc too slow 


    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
        tokens = [
            token.lemma_
            for token in doc
            if token.is_alpha and not token.is_stop
        ]
        tokens_list.append(tokens)
        
        # Verify Lemmatization working 
        # if i == 0:
        #     print("RAW:", texts[i][:300])
        #     print("CLEAN:", df.loc[i, "clean_text"][:300])
        #     print("TOKENS:", tokens)

        if i % 500 == 0:
            print(f"Processed {i} emails...")

    df["tokens"] = tokens_list
    return df
