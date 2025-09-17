# Vehicle Number Plate Detection

Manual identification of vehicle plates is **time-consuming, labor-intensive, and prone to errors**. This project automates the process using **computer vision techniques**, improving efficiency and reliability.
  **Applications include:**
  - 🚓 Law Enforcement  
  - 🅿 Parking Management  
  - 🚧 Toll Collection 

Vehicle number plate detection is a crucial application of computer vision and image processing that focuses on automatically identifying and extracting license plate information from vehicles. Since number plates serve as unique identifiers, this technology plays a vital role in areas such as **intelligent traffic management, law enforcement, toll collection, and parking systems**.

The first step involves cloning the **YOLOv5 model** from **GitHub** and setting it up on the local system. Once the environment is ready, the dataset must be organized by creating separate directories for **training images** and their corresponding **label files**, ensuring that each file is placed in the correct location.

To enhance the dataset quality, **image preprocessing techniques** are applied, including filtering out **blurry images, adjusting brightness levels**, and **improving clarity**. After preprocessing, the training parameters are configured, and a **YAML configuration file** is prepared to define dataset paths and model settings.

The model is then trained by specifying values such as **batch size** and **number of epochs**. Finally, the **trained model** is saved for future use in **vehicle number plate detection tasks**.

In the Additional project I have built an **Retrieval-Augmented Generation (RAG)** system powered by the **Llama 2**. In this system, users can upload PDFs or other documents and interact with it by asking questions. The model first retrieves the most relevant information from the uploaded content and then generates accurate, **context-aware responses**.

This approach enhances the model’s ability to provide **precise answers** that are directly grounded in the user’s data rather than relying solely on pre-trained knowledge.

By using the downloaded **LLM model**, I have create the dataset from the drive folder and apply **text splitter** to divide the content into **chunks** of the defined size. And generate **embedding** using hugging face and store the processed document data in a **vector database**.

Design a **prompt** and set up **answer retrieval pipeline** to generate the final model. And finally, input a question related to the uploaded document and **retrieve the answer from the model’s response**.
