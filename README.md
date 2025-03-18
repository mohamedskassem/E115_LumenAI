<<<<<<< HEAD
## Milestone 3 Template

```
The files are empty placeholders only. You may adjust this template as appropriate for your project.
Never commit large data files,trained models, personal API Keys/secrets to GitHub
```

#### Project Milestone 3 Organization

```
├── Readme.md
├── data # DO NOT UPLOAD DATA TO GITHUB, only .gitkeep to keep the directory or a really small sample
├── midterm_presentation
│   └── CheesyAppMidterm.pdf
├── notebooks
│   └── eda.ipynb
├── references
├── reports
│   └── Statement of Work_Sample.pdf  #This is Milestone1 proposal
└── src
    ├── datapipeline
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── dataloader.py
    │   ├── docker-shell.sh
    │   ├── preprocess_cv.py
    │   ├── preprocess_rag.py
    ├── docker-compose.yml
    └── models
        ├── Dockerfile
        ├── docker-shell.sh
        ├── infer_model.py
        ├── model_rag.py
        └── train_model.py
```

# AC215 - Milestone3 - Cheesy App


**Team Members**
Pavlos Parmigianopapas, Pavlos Ricottapapas and Pavlos Gouda-papas

**Group Name**
The Grate Cheese Group

**Project**
In this project, we aim to develop an AI-powered cheese application. The app will feature visual recognition technology to identify various types of cheese and include a chatbot for answering all kinds of cheese-related questions. Users can simply take a photo of the cheese, and the app will identify it, providing detailed information. Additionally, the chatbot will allow users to ask cheese-related questions. It will be powered by a RAG model and fine-tuned models, making it a specialist in cheese expertise.


<span style="color:red">Our midterm presentation is in the midterm_presentation folder.</span>



----

### Milestone3 ###

In this milestone, we have the components for data management, including versioning, as well as the computer vision and language models.

**Data**
We gathered a dataset of 100,000 cheese images representing approximately 1,500 different varieties. The dataset, approximately 100GB in size, was collected from the following sources: (1), (2), (3). We have stored it in a private Google Cloud Bucket.
Additionally, we compiled 250 bibliographical sources on cheese, including books and reports, from sources such as (4) and (5).

```In this milestone we also added additional preprocessing steps to the data, for better results later on with the fine tuned model.```

**Data Pipeline Containers**
1. One container processes the 100GB dataset by resizing the images and storing them back to Google Cloud Storage (GCS).

	**Input:** Source and destination GCS locations, resizing parameters, and required secrets (provided via Docker).

	**Output:** Resized images stored in the specified GCS location.

2. Another container prepares data for the RAG model, including tasks such as chunking, embedding, and populating the vector database.

## Data Pipeline Overview

1. **`src/datapipeline/preprocess_cv.py`**
   This script handles preprocessing on our 100GB dataset. It reduces the image sizes to 128x128 (a parameter that can be changed later) to enable faster iteration during processing. The preprocessed dataset is now reduced to 10GB and stored on GCS.

2. **`src/datapipeline/preprocess_rag.py`**
   This script prepares the necessary data for setting up our vector database. It performs chunking, embedding, and loads the data into a vector database (ChromaDB).

3. **`src/datapipeline/Pipfile`**
   We used the following packages to help with preprocessing:
   - `special cheese package`

4. **`src/preprocessing/Dockerfile(s)`**
   Our Dockerfiles follow standard conventions, with the exception of some specific modifications described in the Dockerfile/described below.


## Running Dockerfile
Instructions for running the Dockerfile can be added here.
To run Dockerfile - `Instructions here`

**Models container**
- This container has scripts for model training, rag pipeline and inference
- Instructions for running the model container - `Instructions here`

```We worked on improving our model outputs - Details here```

**Midterm Presentation**

Filename: CheesyAppMidterm.pdf


**Notebooks/Reports**
This folder contains code that is not part of container - for e.g: Application mockup, EDA, any 🔍 🕵️‍♀️ 🕵️‍♂️ crucial insights, reports or visualizations.

----
You may adjust this template as appropriate for your project.
=======
# Project Structure

```
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── app/                      # Application implementations
│   │   ├── LumenAI/             # Base OpenAI implementation
│   │   │   ├── app.py           # Main application logic
│   │   │   ├── Dockerfile
│   │   │   ├── Pipfile
│   │   │   ├── dockershell.sh
│   │   │   ├── secrets/         # API keys
│   │   │   └── output/          # Database files
│   │   ├── LumenAI_Advanced/    # Enhanced implementation
│   │   │   ├── app.py
│   │   │   ├── Dockerfile
│   │   │   ├── Pipfile
│   │   │   ├── dockershell.sh
│   │   │   ├── secrets/
│   │   │   └── output/
│   │   ├── LumenAI_Deepseek_Local/  # Local model implementation
│   │   │   ├── app.py
│   │   │   ├── Dockerfile
│   │   │   ├── Pipfile
│   │   │   ├── dockershell.sh
│   │   │   ├── secrets/
│   │   │   └── output/
│   │   └── LumenAI_Mini/        # Lightweight implementation
│   │       ├── app.py
│   │       ├── Dockerfile
│   │       ├── Pipfile
│   │       ├── dockershell.sh
│   │       ├── secrets/
│   │       └── output/
│   ├── data/                    # Data files
│   │   ├── AdventureWorks_Calendar.csv
│   │   ├── AdventureWorks_Customers.csv
│   │   ├── AdventureWorks_Product_Categories.csv
│   │   ├── AdventureWorks_Product_Subcategories.csv
│   │   ├── AdventureWorks_Products.csv
│   │   ├── AdventureWorks_Returns.csv
│   │   ├── AdventureWorks_Sales_2015.csv
│   │   ├── AdventureWorks_Sales_2016.csv
│   │   ├── AdventureWorks_Sales_2017.csv
│   │   └── AdventureWorks_Territories.csv
│   └── data_loader/            # Data loading utilities
│       ├── Dockerfile
│       ├── load_data.py
│       ├── Pipfile
│       └── Pipfile.lock
└── reports/                    # Project documentation
    ├── M2_Report.pdf
    └── Statement_of_Work.pdf
```

### Application Variants

Each application variant follows a similar structure with its own specific implementation:

**1. LumenAI (Base Implementation)**
- Core Text-to-SQL functionality using OpenAI's API
- Features:
  - Natural language to SQL conversion
  - Schema-aware query generation
  - Interactive CLI interface
  - Vector-based schema understanding
- Dependencies: OpenAI API, LlamaIndex, HuggingFace embeddings

**2. LumenAI_Advanced**
- Enhanced version of the base implementation
- Additional features:
  - Advanced query optimization
  - More sophisticated schema analysis
  - Extended error handling
  - Improved result analysis
- Dependencies: Same as base + additional optimization libraries

**3. LumenAI_Deepseek_Local**
- Locally running implementation using Deepseek models
- Features:
  - No dependency on external API
  - Local model inference
  - Privacy-focused implementation
  - Suitable for offline use
- Dependencies: Deepseek models, local inference requirements

**4. LumenAI_Mini**
- Lightweight implementation for resource-constrained environments
- Features:
  - Reduced model size
  - Optimized for performance
  - Minimal dependencies
  - Faster query processing
- Dependencies: Minimized version of base requirements

### Common Components Across All Variants

Each application variant includes:

```
.
├── app.py              # Main application logic
├── Dockerfile         # Container configuration
├── Pipfile           # Python dependencies
├── dockershell.sh    # Docker run script
├── secrets/          # API keys and credentials
└── output/           # Database and output files
```

### Running Instructions

Each variant can be run using Docker:
```bash
cd src/app/<variant_folder>
./dockershell.sh
```

### Requirements
- Docker
- Python 3.9+
- Relevant API keys (for OpenAI variant)
- SQLite database
- Model files (for local variants)


  
>>>>>>> milestone2
