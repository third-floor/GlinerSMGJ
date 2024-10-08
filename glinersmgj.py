import streamlit as st
from gliner import GLiNER
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

# Cache the download of nltk resources
@st.cache_data
def download_nltk_resources():
    nltk.download('punkt')

download_nltk_resources()

# Cache the GLiNER model loading to avoid reloading on every app rerun
@st.cache_resource
def load_model():
    return GLiNER.from_pretrained("urchade/gliner_base")

model = load_model()

# Function to chunk text by sentences (preserving names and avoiding token breakups)
def chunk_text_by_sentences(text, max_length=512):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Function to process a batch of chunks
def process_batch(chunks, labels):
    try:
        batch_text = " ".join(chunks)
        return model.predict_entities(batch_text, labels)
    except Exception as e:
        print(f"Error processing batch: {e}")
        return []

# Function to post-process entities (remove duplicates or merge adjacent entities if needed)
def post_process_entities(entities):
    processed = []
    seen = set()

    for entity in entities:
        entity_text = entity['text']
        if entity_text not in seen:
            processed.append(entity)
            seen.add(entity_text)

    return processed

# Function to annotate the text with entity labels
def annotate_text_with_entities(text, entities):
    annotated_text = text
    sorted_entities = sorted(entities, key=lambda x: text.find(x["text"]), reverse=True)

    for entity in sorted_entities:
        annotated_text = annotated_text.replace(entity["text"], f'{entity["text"]} [{entity["label"]}]')

    return annotated_text

# Function to handle entity extraction from long text
def extract_entities(text, labels, batch_size=3, max_workers=4):
    # Split text into manageable chunks by sentences
    chunks = chunk_text_by_sentences(text)

    all_entities = []
    
    # Process chunks in parallel using ThreadPoolExecutor with batch size
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            futures.append(executor.submit(process_batch, batch, labels))

        for future in as_completed(futures):
            try:
                entities = future.result()
                all_entities.extend(entities)
            except Exception as e:
                print(f"Error in thread: {e}")

    # Post-process entities to remove duplicates or merge adjacent ones
    return post_process_entities(all_entities)

# Function to count the number of different entities for each class
def count_entities_by_class(entities):
    entity_counts = defaultdict(int)

    for entity in entities:
        entity_counts[entity['label']] += 1

    return entity_counts

# Function to list all unique entities grouped by their types
def list_unique_entities_by_type(entities):
    unique_entities_by_type = defaultdict(set)
    
    for entity in entities:
        unique_entities_by_type[entity['label']].add(entity["text"])  # Group by label

    return unique_entities_by_type

# Streamlit UI starts here
st.title("Entity Recognition App")

# Input for text
input_text = st.text_area("Enter the text to process", height=200)

# Input for entity labels
st.write("Enter up to 5 entity labels:")
labels = [st.text_input(f"Label {i+1}", value="") for i in range(5)]

# Button to submit text and labels
if st.button("Submit"):
    # Filter out empty labels
    labels = [label for label in labels if label.strip()]

    if input_text and labels:
        st.write("Processing...")

        try:
            # Extract entities
            entities = extract_entities(input_text, labels)
            
            # Annotate text
            annotated_text = annotate_text_with_entities(input_text, entities)
            st.subheader("Annotated Text:")
            st.write(annotated_text)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please provide both text and at least one entity label.")
