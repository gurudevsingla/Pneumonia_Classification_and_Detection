<<<<<<< HEAD
# Business Problem:

- Pneumonia is a serious transmittable disease in India, It is responsible for high morbidity and mortality amongst children under five year of age. India accounts for one-third of the total WHO South East Asia burden of under-five mortality. (source: https://pmc.ncbi.nlm.nih.gov/articles/PMC6131850/).
- Chest X-rays are currently the best available method for diagnosing it.
- Suppose You are working as a Data Scientist at Qure.ai (a Medical startup) and want to Classify if a person has pneumonia or not.
  - You also have to deploy the model on mobile device for real time inferences

  - Please go through this link to know about the existing apps in medical domain :
  
      https://www.grantsformedical.com/apps-for-medical-diagnosis.html

- This was especially useful during the times when COVID-19 was known to cause pneumonia.


<img src='https://drive.google.com/uc?id=1Qc7aF9zBzFz-I6lnhXZK1IOQjwLTfET5' width=400>


   - Image on the left is a normal,
    but on the right we can see severe glass opacity mainly due to air displacement by fluids

## Brief intro:

What is Pneumonia?

- Pneumonia is an infection that inflames the air sacs in one or both lungs.
- The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing.
- A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia.

- Pneumonia can range in seriousness from mild to life-threatening.
- It is most serious for infants and young children, people older than age 65, and people with health problems or weakened immune systems.

<img src='https://drive.google.com/uc?id=1I7HBz2uVPq8SOcjC7el86EZR8kfxy6aK' width=300>

### Real time constraints:
- Low latency requirements
- False negatives or positives can be risky
- Model should be confident in deciding the class
- Model explainability through visualizations

# Data

Dataset is present inside tensorflow.data.TFRecordDataset

# Approach

- State-of-the-art models like VGG16, InceptionNet, ResNet, EfficientNet etc.are computationally heavy and cannot be deployed on mobile devices.

- So we are going to use mobilenet
    - MobileNet achieves sky high accuracies and is 100x smaller compared to these models
    - MobileNet is used on mobile apps for object detection, image classficiation, etc. and provide low latency outputs for any use case.
    

<center><img src='https://drive.google.com/uc?id=10Do0zifY8E3gyyPSZQQsMyiOrzc92FSn' width=700>

MobileNet Architecture</center>

Paper Link - https://arxiv.org/abs/1704.04861

- Finetuning MobineNetV2 for classifying images.
- There are way more images that are classified as pneumonia than normal in our dataset.
- This shows that we have an **imbalance** in our data i.e our model can be biased towards high majority class (Pneumonia in our case)

### **How we are going to solve the class imbalance here ?**

- Here Data augmentation will not be useful because X-ray scans are only taken in a specific orientation, and variations such as flips and rotations will not exist in real X-ray images.

- We will correct for this imbalance later using class weights.

### **Figuring out which part of the image our model focuses on to get a prediction ?**

* We, **as humans**, when we try to classify an image.
* **we look at certain region in the image to make our judgement**.
* **For eg:** In case of cats & dog classification, we’ll focus on face of the animal in the picture.
* In this case, i.e. pneunomnia classification, **the domain expert or doctor look at the specific region of the xray in the frame**.

**How can we check what region in the image our CNN is focusing on ?**

* With the help of **GradCAM(Gradient-weighted Class Activation Map**) algorithm.
* It helps to find out the region on which **CNN is focusing on to predict particular class**.

Paper Link - https://arxiv.org/pdf/1610.02391

### Final Results

- **Recall** : 0.971794843673706
- **Precision**: 0.7734693884849548

# Deployement on Streamlit
Link - https://pneumonia-classification-and-detection.streamlit.app/
"""
=======
"# Business Problem:\n",
        "\n",
        "- Pneumonia is a serious transmittable disease in India, It is responsible for high morbidity and mortality amongst children under five year of age. India accounts for one-third of the total WHO South East Asia burden of under-five mortality. (source: https://pmc.ncbi.nlm.nih.gov/articles/PMC6131850/).\n",
        "- Chest X-rays are currently the best available method for diagnosing it.\n",
        "- Suppose You are working as a Data Scientist at Qure.ai (a Medical startup) and want to Classify if a person has pneumonia or not.\n",
        "  - You also have to deploy the model on mobile device for real time inferences\n",
        "\n",
        "  - Please go through this link to know about the existing apps in medical domain :\n",
        "  \n",
        "      https://www.grantsformedical.com/apps-for-medical-diagnosis.html\n",
        "\n",
        "- This was especially useful during the times when COVID-19 was known to cause pneumonia.\n",
        "\n",
        "\n",
        "<img src='https://drive.google.com/uc?id=1Qc7aF9zBzFz-I6lnhXZK1IOQjwLTfET5' width=400>\n",
        "\n",
        "\n",
        "   - Image on the left is a normal,\n",
        "    but on the right we can see severe glass opacity mainly due to air displacement by fluids\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "6E309wiYwA7n"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Brief intro:\n",
        "\n",
        "What is Pneumonia?\n",
        "\n",
        "- Pneumonia is an infection that inflames the air sacs in one or both lungs.\n",
        "- The air sacs may fill with fluid or pus (purulent material), causing cough with phlegm or pus, fever, chills, and difficulty breathing.\n",
        "- A variety of organisms, including bacteria, viruses and fungi, can cause pneumonia.\n",
        "\n",
        "- Pneumonia can range in seriousness from mild to life-threatening.\n",
        "- It is most serious for infants and young children, people older than age 65, and people with health problems or weakened immune systems.\n",
        "\n",
        "<img src='https://drive.google.com/uc?id=1I7HBz2uVPq8SOcjC7el86EZR8kfxy6aK' width=300>"
      ],
      "metadata": {
        "id": "q8nmo0PYxEdI"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Real time constraints:\n",
        "- Low latency requirements\n",
        "- False negatives or positives can be risky\n",
        "- Model should be confident in deciding the class\n",
        "- Model explainability through visualizations"
      ],
      "metadata": {
        "id": "AQ7eKNN5xX34"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Data\n",
        "\n",
        "Dataset is present inside tensorflow.data.TFRecordDataset"
      ],
      "metadata": {
        "id": "LpFIglVCxoz0"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Approach\n",
        "\n",
        "- State-of-the-art models like VGG16, InceptionNet, ResNet, EfficientNet etc.are computationally heavy and cannot be deployed on mobile devices.\n",
        "\n",
        "- So we are going to use mobilenet\n",
        "    - MobileNet achieves sky high accuracies and is 100x smaller compared to these models\n",
        "    - MobileNet is used on mobile apps for object detection, image classficiation, etc. and provide low latency outputs for any use case.\n",
        "    \n",
        "\n",
        "<center><img src='https://drive.google.com/uc?id=10Do0zifY8E3gyyPSZQQsMyiOrzc92FSn' width=700>\n",
        "\n",
        "MobileNet Architecture</center>\n",
        "\n",
        "Paper Link - https://arxiv.org/abs/1704.04861\n",
        "\n",
        "- Finetuning MobineNetV2 for classifying images.\n",
        "- There are way more images that are classified as pneumonia than normal in our dataset.\n",
        "- This shows that we have an **imbalance** in our data i.e our model can be biased towards high majority class (Pneumonia in our case)\n",
        "\n",
        "### **How we are going to solve the class imbalance here ?**\n",
        "\n",
        "- Here Data augmentation will not be useful because X-ray scans are only taken in a specific orientation, and variations such as flips and rotations will not exist in real X-ray images.\n",
        "\n",
        "- We will correct for this imbalance later using class weights.\n",
        "\n",
        "### **Figuring out which part of the image our model focuses on to get a prediction ?**\n",
        "\n",
        "* We, **as humans**, when we try to classify an image.\n",
        "* **we look at certain region in the image to make our judgement**.\n",
        "* **For eg:** In case of cats & dog classification, we’ll focus on face of the animal in the picture.\n",
        "* In this case, i.e. pneunomnia classification, **the domain expert or doctor look at the specific region of the xray in the frame**.\n",
        "\n",
        "**How can we check what region in the image our CNN is focusing on ?**\n",
        "\n",
        "* With the help of **GradCAM(Gradient-weighted Class Activation Map**) algorithm.\n",
        "* It helps to find out the region on which **CNN is focusing on to predict particular class**.\n",
        "\n",
        "Paper Link - https://arxiv.org/pdf/1610.02391\n",
        "\n",
        "### Final Results\n",
        "\n",
        "- **Recall** : 0.971794843673706\n",
        "- **Precision**: 0.7734693884849548"
      ],
      "metadata": {
        "id": "LgXD7PbhyWtx"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Deployement on Streamlit\n",
        "Link - https://pneumonia-classification-and-detection.streamlit.app/"
      ],
      "metadata": {
        "id": "wYLwtQYty_hQ"
      }
    }
  ]
}
>>>>>>> c253236d33c0b7ee06df6d319cfc1605c9670029
