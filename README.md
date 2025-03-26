# Project Structure

```
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── app/                      # Application implementation
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   ├── Pipfile
│   │   ├── dockershell.sh
│   │   ├── secrets/
│   │   └── output/
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

The project now focuses on a single, optimized Text-to-SQL system with the following enhancements since Milestone 2:

**Key Enhancements**
- **Conversation History**: Maintains chat context for improved follow-up queries
- **Parallel Processing**: Multi-threaded table and column analysis for faster loading
- **Performance Optimization**: Schema analysis caching for quicker startup times
- **Contextual SQL Generation**: Generates SQL queries using both schema and conversation history
- **Vector-based Schema Understanding**: Leverages LlamaIndex embeddings for semantic understanding of database structure
- **Comparative Analysis**: Results analysis now includes comparisons with previous query results
- **Optimized Prompts**: More concise and effective prompts for better LLM responses

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
cd src/app
./dockershell.sh
```

### Requirements
- Docker
- Python 3.9+
- OpenAI API key
- SQLite database
