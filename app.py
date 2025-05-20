import streamlit as st
from tokenizer import CharTokenizer
from model import Generator

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(page_title="Text Generator", layout="centered")

# ----------------------
# Title & Description
# ----------------------
st.title("Text Generator")
st.markdown(
    """
    Generate text character-by-character using a trained LSTM language model.
    
    Enter a prompt and choose how many characters you'd like to generate!
    """
)

# ----------------------
# Load Model & Tokenizer (Cached)
# ----------------------
@st.cache_resource
def load_generator_and_tokenizer():
    try:
        tokenizer = CharTokenizer.load('token.json')
        generator = Generator.load_from_checkpoint(
            "/Users/siddha-book/Desktop/Project/Python Projects/Text Generator/lightning_logs/version_4/checkpoints/epoch=0-step=63.ckpt",
            tokenizer=tokenizer
        )
        return tokenizer, generator
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None


tokenizer, generator = load_generator_and_tokenizer()

if tokenizer is None or generator is None:
    st.stop()

# ----------------------
# Input Form
# ----------------------
with st.form("generate_form"):
    st.subheader("📝 Input Prompt")
    
    col1, col2 = st.columns([3, 1])

    with col1:
        prompt = st.text_input("Enter your starting text:", value="Once upon a", key="prompt_input")

    with col2:
        n_token = st.number_input("Characters to generate:", min_value=10, max_value=500, value=200, step=10)

    submit_button = st.form_submit_button("🚀 Generate Text", use_container_width=True)

# ----------------------
# Generation Logic
# ----------------------
if submit_button:
    if not prompt.strip():
        st.warning("Please enter a valid prompt.")
    else:
        with st.spinner("🧠 Generating text..."):
            output = generator.generate(prompt.strip(), n_token=n_token)
        
        # ----------------------
        # Display Output
        # ----------------------
        st.subheader("🤖 Generated Text")
        # st.markdown(
        #     f"""
        #     <div style="
        #         background-color: #f0f2f6;
        #         padding: 15px;
        #         border-radius: 8px;
        #         font-family: monospace;
        #         white-space: pre-wrap;
        #         word-wrap: break-word;
        #         border: 1px solid #ccc;
        #     ">
        #         {output}
        #     </div>
        #     """,
        #     unsafe_allow_html=True,
        # )

        # Optional: Show copy button
        st.code(output)  # Simple alternative for copy-paste

# # ----------------------
# # Footer
# # ----------------------
# st.markdown("---")
# st.markdown("<p style='text-align:center;'>Built with ❤️ using Streamlit & PyTorch Lightning</p>", unsafe_allow_html=True)