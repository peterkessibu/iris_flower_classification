# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image  # For handling image uploads
import os  # For working with file directories

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f8f9fa;
    }
    .main-header {
        background-color: #4CAF50;
        color: white;
        padding: 15px;
        text-align: center;
        border-radius: 10px;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .upload-area {
        border: 2px dashed #4CAF50;
        padding: 20px;
        border-radius: 10px;
        background-color: #e8f5e9;
        text-align: center;
    }
    .upload-area:hover {
        background-color: #d7ffd9;
    }
    .btn {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 5px;
        cursor: pointer;
    }
    .btn:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit App Title
st.markdown('<div class="main-header"><h1>Iris Species Classification</h1></div>', unsafe_allow_html=True)

# Load the iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels (species)
df = pd.DataFrame(data=X, columns=iris.feature_names)
df['species'] = y

# Display the DataFrame in a styled card
st.markdown('<div class="card"><h2>Iris Dataset Preview:</h2></div>', unsafe_allow_html=True)
st.dataframe(df.head())

# Sidebar for model configuration
st.sidebar.header("Model Configuration")
max_depth = st.sidebar.slider("Max Depth of Decision Tree:", 1, 10, 3)
criterion = st.sidebar.selectbox("Criterion:", ("gini", "entropy"))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
model.fit(X_train, y_train)

# Test the model and calculate accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
st.markdown('<div class="card"><h2>Model Performance</h2>', unsafe_allow_html=True)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Display the confusion matrix
st.markdown('<div class="card"><h2>Confusion Matrix:</h2></div>', unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
st.pyplot(fig)

# Upload space for images
st.markdown('<div class="card"><h2>Upload an Image</h2></div>', unsafe_allow_html=True)
st.markdown('<div class="upload-area">Drop your image here or click to upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Image uploaded successfully!")

# Display public images
st.markdown('<div class="card"><h2>Public Images</h2></div>', unsafe_allow_html=True)
public_folder = "public_images"  # Folder containing public images
if not os.path.exists(public_folder):
    os.makedirs(public_folder)  # Create folder if it doesn't exist
    # Optionally, add some default images to the folder for testing

# Display all images in the public folder
public_images = [f for f in os.listdir(public_folder) if f.endswith(("png", "jpg", "jpeg"))]
if public_images:
    for image_file in public_images:
        image_path = os.path.join(public_folder, image_file)
        image = Image.open(image_path)
        st.image(image, caption=image_file, use_column_width=True)
else:
    st.write("No public images found. Add images to the 'public_images' folder.")
