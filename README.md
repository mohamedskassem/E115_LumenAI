# Project Structure

```
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── app/                      # Application implementations
│   │   └── LumenAI_Advanced/    # Enhanced implementation
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
    └── M2_Report.pdf
```

### Application Implementation

The project uses LumenAI_Advanced, which is an enhanced implementation of the Text-to-SQL system:

**LumenAI_Advanced**
- Enhanced implementation with advanced features
- Features:
  - Natural language to SQL conversion
  - Schema-aware query generation
  - Interactive CLI interface
  - Vector-based schema understanding
  - Advanced query optimization
  - Sophisticated schema analysis
  - Extended error handling
  - Improved result analysis
- Dependencies: 
  - OpenAI API
  - LlamaIndex
  - HuggingFace embeddings
  - Additional optimization libraries

### Components Structure

The application includes:

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

To run the application using Docker:
```bash
cd src/app/LumenAI_Advanced
./dockershell.sh
```

### Requirements
- Docker
- Python 3.9+
- OpenAI API key
- SQLite database
