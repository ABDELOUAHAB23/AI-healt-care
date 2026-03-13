import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import plotly.graph_objects as go
from model import DenseNet121, CLASS_NAMES, set_seed
import io
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Chest X-Ray Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better design
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .uploadedFile {
        border: 2px dashed #4CAF50;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stProgress {
        height: 20px;
        border-radius: 10px;
    }
    h1 {
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    h2 {
        color: #34495e;
        margin: 1.5rem 0;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def load_model():
    """Load the pre-trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(len(CLASS_NAMES)).to(device)
    
    try:
        checkpoint = torch.load('model.pth.tar', map_location=device)
        state_dict = checkpoint['state_dict']
        if not list(state_dict.keys())[0].startswith('module.'):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, device

def predict_image(model, image, device):
    """Make prediction for an image"""
    # Image preprocessing
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    
    # Transform and predict
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    
    probabilities = output.cpu().numpy()[0]
    return probabilities

def create_prediction_plot(predictions, names):
    """Create a horizontal bar plot for predictions"""
    fig = go.Figure(go.Bar(
        x=predictions * 100,  # Convert to percentages
        y=names,
        orientation='h',
        marker=dict(
            color=predictions * 100,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Probability (%)")
        )
    ))
    
    fig.update_layout(
        title="Disease Probability Analysis",
        xaxis_title="Probability (%)",
        yaxis_title="Conditions",
        height=600,
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 100])
    )
    
    return fig

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Header
    st.title("🏥 Chest X-Ray Analysis System")
    st.markdown("""
    This system uses deep learning to analyze chest X-ray images and detect potential medical conditions.
    Upload an X-ray image to get started.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This system can detect 14 different medical conditions from chest X-rays using
        a deep learning model based on DenseNet121 architecture.
        
        **Supported Conditions:**
        """)
        for condition in CLASS_NAMES:
            st.markdown(f"- {condition}")
            
        st.markdown("""
        ---
        ⚠️ **Disclaimer:** This tool is for educational purposes only.
        Always consult healthcare professionals for medical diagnosis.
        """)
    
    # Load model
    model, device = load_model()
    if model is None:
        st.error("Failed to load the model. Please check if model file exists.")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Upload a chest X-ray image (PNG, JPG, JPEG)", 
                                   type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        try:
            # Display the uploaded image
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.subheader("Uploaded X-Ray")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Analysis Results")
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    probabilities = predict_image(model, image, device)
                    
                    # Sort predictions
                    sorted_idx = np.argsort(probabilities)[::-1]
                    sorted_probs = probabilities[sorted_idx]
                    sorted_names = [CLASS_NAMES[i] for i in sorted_idx]
                    
                    # Create and display plot
                    fig = create_prediction_plot(sorted_probs, sorted_names)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display top 3 findings
                    st.markdown("### Key Findings")
                    for i in range(3):
                        if sorted_probs[i] > 0.5:  # Only show if probability > 50%
                            st.markdown(f"""
                            <div class='prediction-box'>
                                <b>{sorted_names[i]}:</b> {sorted_probs[i]*100:.1f}% probability
                            </div>
                            """, unsafe_allow_html=True)
            
            # Disclaimer
            st.markdown("""
            ---
            **Note:** These results are computer-generated predictions and should not be used as a substitute
            for professional medical diagnosis. Please consult with healthcare professionals for proper
            medical advice and treatment.
            """)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
