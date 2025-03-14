from flask import Flask, request, jsonify, render_template
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

app = Flask(__name__)

# Expanded Personal Information Document
documents = ["""
My name is Vuong Loc Truong, and I am a Vietnamese citizen. I was born on October 20, 2001, and I am 23 years old. I am currently pursuing a Master's degree in Data Science and AI, which is my highest level of education thus far. My academic journey has focused on the intricacies of data and artificial intelligence, a field I find both challenging and profoundly rewarding. While I currently have no formal work experience, my involvement in web development has provided me with practical insights into the technological landscape. Presently, I serve as a teaching assistant at Van Lang University, where I contribute to the educational development of students.
             
My core belief regarding the role of technology in shaping society is that it holds immense potential to enhance the quality of life. By providing solutions to complex problems in healthcare, education, and environmental sustainability, technology can be a powerful force for good. However, I also recognize the importance of ensuring equitable access and addressing ethical concerns to prevent the exacerbation of existing inequalities. Responsible innovation and thoughtful regulation are crucial for harnessing technology’s power for the greater good.

Furthermore, I believe that cultural values should significantly influence technological advancements. Technology should be developed and implemented in a manner that respects and preserves diverse cultural identities and traditions. Avoiding the imposition of a single cultural perspective is essential. Instead, prioritizing inclusivity and adaptability to different cultural contexts ensures that technology serves the needs of diverse communities and promotes cultural understanding.

As a Master's student, I find that the most challenging aspect of my studies thus far is English communication. Overcoming this obstacle is a priority for me. My primary academic goal is to graduate on time, ensuring that I can effectively apply my knowledge and contribute to the field of Data Science and AI. I am dedicated to my studies and eager to see how my research interests will evolve and contribute to the technological advancements of the future.

In addition to my academic pursuits, I have a strong interest in web development. This interest has led me to work on various projects, both personal and academic, that involve creating and maintaining websites. My experience in web development has provided me with a solid understanding of front-end and back-end technologies, as well as the importance of user experience and accessibility. I believe that web development is a crucial skill in today's digital age, and I am committed to continuing my growth in this area.

At Van Lang University, I have had the opportunity to work closely with students as a teaching assistant. This role has allowed me to develop my communication and mentoring skills, as well as gain a deeper understanding of the educational process. I take great pride in helping students achieve their academic goals and am always looking for ways to improve my teaching methods.

Looking ahead, I am excited about the potential for technology to drive positive change in society. I am particularly interested in exploring how data science and artificial intelligence can be used to address pressing global challenges, such as climate change, healthcare, and education. I am committed to using my skills and knowledge to contribute to these efforts and to make a meaningful impact on the world.
"""]

# Alternative: Using SentenceTransformers to generate embeddings locally (no API limit)
def create_embeddings_with_sentence_transformer(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model
    embeddings = model.encode(documents)
    return embeddings

# Create embeddings using SentenceTransformer embeddings (offline)
embeddings = create_embeddings_with_sentence_transformer(documents)

# Convert the embeddings to a numpy array (required by FAISS)
embeddings = np.array(embeddings).astype("float32")

# Create the FAISS index
dimension = embeddings.shape[1]  # The dimensionality of the embeddings
index = faiss.IndexFlatL2(dimension)  # Use L2 distance for similarity search

# Add the embeddings to the index
index.add(embeddings)

# Create the docstore and the index_to_docstore_id dictionary
index_to_docstore_id = {i: str(i) for i in range(len(documents))}
docstore = InMemoryDocstore({index_to_docstore_id[i]: Document(page_content=doc) for i, doc in enumerate(documents)})

# Create the FAISS retriever manually using the index and other required arguments
db = FAISS(index=index, embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"), docstore=docstore, index_to_docstore_id=index_to_docstore_id)

# Use FAISS to store embeddings and enable retrieval
retriever = db.as_retriever()

# Load the model and tokenizer for QA
qa_model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
qa_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')

# Define QA pipeline
qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.get_json()
        question = user_message['question'] if user_message else None
        if not question:
            return jsonify({"error": "No question provided"}), 400
        response = generate_answer(question)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_answer(question):
    retrieved_docs = retriever.get_relevant_documents(question)
    context = " ".join([doc.page_content for doc in retrieved_docs])
    result = qa_pipeline(question=question, context=context)
    answer = result['answer']
    return {"answer": answer, "context": context}

if __name__ == '__main__':
    app.run(debug=True)