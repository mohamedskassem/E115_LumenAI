# Project Structure

├── README.md
├── data/ 
│ ├── AdventureWorks_Calendar.csv # Date records (2015-2017)
│ ├── AdventureWorks_Customers.csv # Customer information (1.9MB)
│ ├── AdventureWorks_Product_Categories.csv # Product category listings (83B)
│ ├── AdventureWorks_Product_Subcategories.csv # Product subcategories (637B)
│ ├── AdventureWorks_Products.csv # Product details (57KB)
│ ├── AdventureWorks_Returns.csv # Product returns data (34KB)
│ ├── AdventureWorks_Sales_2015.csv # Sales records for 2015 (116KB)
│ ├── AdventureWorks_Sales_2016.csv # Sales records for 2016 (1.0MB)
│ ├── AdventureWorks_Sales_2017.csv # Sales records for 2017 (1.3MB)
│ └── AdventureWorks_Territories.csv # Sales territories (400B)
├── reports/
│ ├── M2_Report.pdf
│ └── Statement_of_Work.pdf
└── src/
├── app/ # Application implementations
│ ├── LumenAI/ # Base OpenAI implementation
│ │ ├── app.py # Main application logic
│ │ ├── Dockerfile
│ │ ├── Pipfile
│ │ ├── dockershell.sh
│ │ ├── secrets/ # API keys
│ │ └── output/ # Database files
│ ├── LumenAI_Advanced/ # Enhanced implementation
│ │ ├── app.py
│ │ ├── Dockerfile
│ │ ├── Pipfile
│ │ ├── dockershell.sh
│ │ ├── secrets/
│ │ └── output/
│ ├── LumenAI_Deepseek_Local/ # Local model implementation
│ │ ├── app.py
│ │ ├── Dockerfile
│ │ ├── Pipfile
│ │ ├── dockershell.sh
│ │ ├── secrets/
│ │ └── output/
│ └── LumenAI_Mini/ # Lightweight implementation
│ ├── app.py
│ ├── Dockerfile
│ ├── Pipfile
│ ├── dockershell.sh
│ ├── secrets/
│ └── output/
├── data_loader/
| ├── Dockerfile
| ├── load_data.py
| ├── Pipfile
| └── Pipfile.lock


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

├── app.py              # Main application logic
├── Dockerfile         # Container configuration
├── Pipfile           # Python dependencies
├── dockershell.sh    # Docker run script
├── secrets/          # API keys and credentials
└── output/           # Database and output files

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


  