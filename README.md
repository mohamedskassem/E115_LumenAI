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


  