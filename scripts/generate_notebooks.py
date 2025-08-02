import os
import nbformat as nbf

# Ordner anlegen
os.makedirs('notebooks', exist_ok=True)

# Notebooks-Definition: (Dateiname, Liste von Zellen)
notebooks = [
    ("00_data_overview.ipynb", [
        nbf.v4.new_markdown_cell("# Data Overview\n\n**Ziel:** Erste Einblicke in die Rohdaten"),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "df = pd.read_csv('../data/user_tool_interactions.csv')\n"
            "df.head()\n"
            "df.info()\n"
            "df.describe()"
        )
    ]),
    ("01_matrix_creation.ipynb", [
        nbf.v4.new_markdown_cell("# Matrix Creation\n\n**Ziel:** User-Tool-Matrix erstellen"),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "df = pd.read_csv('../data/user_tool_interactions.csv')\n"
            "R_df = df.pivot_table(index='user_id', columns='tool_id', values='usage_count', fill_value=0)\n"
            "display(R_df.head())\n"
            "plt.imshow(R_df, aspect='auto')\n"
            "plt.title('Heatmap User-Tool Matrix')\n"
            "plt.show()"
        )
    ]),
    ("02_user_similarity_heatmap.ipynb", [
        nbf.v4.new_markdown_cell("# User Similarity Heatmap\n\n**Ziel:** Cosine-Similarity visualisieren"),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "from sklearn.metrics.pairwise import cosine_similarity\n"
            "import matplotlib.pyplot as plt\n\n"
            "df = pd.read_csv('../data/user_tool_interactions.csv')\n"
            "R_df = df.pivot_table(index='user_id', columns='tool_id', values='usage_count', fill_value=0)\n"
            "S = cosine_similarity(R_df)\n"
            "plt.imshow(S, aspect='auto')\n"
            "plt.colorbar(label='Similarity')\n"
            "plt.title('User Similarity')\n"
            "plt.show()"
        )
    ]),
    ("03_recommendation_demo.ipynb", [
        nbf.v4.new_markdown_cell("# Recommendation Demo\n\n**Ziel:** Beispiel-Empfehlungen abrufen"),
        nbf.v4.new_code_cell(
            "import sys\n"
            "sys.path.append('..')\n"
            "import pandas as pd\n"
            "from src.recommender import CosineRecommender, build_interaction_matrix\n\n"
            "df = pd.read_csv('../data/user_tool_interactions.csv')\n"
            "R, users, tools = build_interaction_matrix(df)\n"
            "rec = CosineRecommender(R, users, tools)\n"
            "print(rec.recommend('U1', top_k=5))"
        )
    ]),
    ("04_evaluation.ipynb", [
        nbf.v4.new_markdown_cell("# Evaluation\n\n**Ziel:** Precision@K, Recall@K messen"),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "from sklearn.model_selection import train_test_split\n"
            "from src.recommender import CosineRecommender, build_interaction_matrix\n\n"
            "df = pd.read_csv('../data/user_tool_interactions.csv')\n"
            "train, test = train_test_split(df, test_size=0.2, random_state=42)\n"
            "R_train, users, tools = build_interaction_matrix(train)\n"
            "rec = CosineRecommender(R_train, users, tools)\n\n"
            "# Hier Precision@K / Recall@K implementieren..."
        )
    ]),
    ("05_api_playground.ipynb", [
        nbf.v4.new_markdown_cell("# API Playground\n\n**Ziel:** FastAPI-Endpoint im Notebook testen"),
        nbf.v4.new_code_cell(
            "import requests\n"
            "import pandas as pd\n\n"
            "API = 'http://localhost:8000'\n"
            "res = requests.get(f'{API}/recommend/U1?k=5')\n"
            "pd.DataFrame(res.json()['recommendations'])"
        )
    ]),
]

for fname, cells in notebooks:
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            }
        }
    )
    path = os.path.join('notebooks', fname)
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

print("Notebooks wurden angelegt unter 'notebooks/'.")
